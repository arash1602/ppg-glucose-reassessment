// PPG -> glucose firmware for ESP32-S3, STM32F746G-DISCO and NUCLEO-WB55 (same C model code on all three)
//  Boot: self-test of the device model (PPG features + gradient-boosted trees) on real VitalDB
//        windows against the Python results, plus timing of the per-beat CNN where it is built in.
//  Then: command protocol over the USB serial port (see device.h):
//        PROFILE <age> <sex_m> <bmi> <diabetes> | CAL <mg/dL> | MIN <minutes> | RESET | BEGIN <n> ... END
//        s            re-run the self-test
//        BENCH <s>    energy benchmark: <s> seconds idle, then <s> seconds of back-to-back estimates
#include <Arduino.h>
#include <stdarg.h>
#include <string.h>
#include "glucose.h"
#include "device.h"
#ifndef GLU_NO_CNN
#include "test_window.h"
#endif
#include "dev_test.h"

#if defined(ARDUINO_ARCH_ESP32)
// the default 8 kB loop-task stack is too tight for feature extraction + JSON output; take 16 kB from the heap
SET_LOOP_TASK_STACK_SIZE(16 * 1024);
#define BOARD_NAME "ESP32-S3"
#elif defined(ARDUINO_ARCH_STM32)
#if defined(STM32F7xx)
#define BOARD_NAME "STM32F746G-DISCO"
#elif defined(STM32WBxx)
#define BOARD_NAME "NUCLEO-WB55"
#else
#define BOARD_NAME "STM32"
#endif
#endif

#ifndef SERIAL_BAUD
#define SERIAL_BAUD 921600
#endif

static uint32_t micros_hook(void) { return (uint32_t)micros(); }
static double minutes_hook(void) { return millis() / 60000.0; }
static void emit(const char *l) { Serial.println(l); }

static void outf(const char *fmt, ...) {          // printf that works on every Arduino core
    static char buf[256];
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(buf, sizeof buf, fmt, ap);
    va_end(ap);
    Serial.print(buf);
}

static unsigned free_ram_kb(void) {
#if defined(ARDUINO_ARCH_ESP32)
    return ESP.getFreeHeap() / 1024;
#elif defined(ARDUINO_ARCH_STM32)
    char top;
    char *heap_end = (char *)malloc(16);              // current end of the heap
    unsigned kb = heap_end ? (unsigned)((&top - heap_end) / 1024) : 0;
    free(heap_end);
    return kb;                                        // gap between heap end and stack
#else
    return 0;
#endif
}

static void board_info(void) {
#if defined(ARDUINO_ARCH_ESP32)
    outf("board %s | chip %s rev %d, %d cores @ %d MHz | flash %u kB, PSRAM %u kB | sketch %u kB | free heap %u kB\n",
         BOARD_NAME, ESP.getChipModel(), ESP.getChipRevision(), ESP.getChipCores(), ESP.getCpuFreqMHz(),
         ESP.getFlashChipSize() / 1024, ESP.getPsramSize() / 1024, ESP.getSketchSize() / 1024, ESP.getFreeHeap() / 1024);
#else
    outf("board %s | core clock %lu MHz | free RAM (heap-stack gap) %u kB\n", BOARD_NAME,
         (unsigned long)(SystemCoreClock / 1000000UL), free_ram_kb());
#endif
}

static dev_features_t g_f;
#ifndef GLU_NO_CNN
static glu_result_t g_res;
#endif

static bool check(const char *name, double dev, double py, double tol) {
    bool ok = fabs(dev - py) <= tol;
    outf("  %-34s device %.6f  Python %.6f  diff %.2e  %s\n", name, dev, py, fabs(dev - py), ok ? "PASS" : "FAIL");
    return ok;
}

static void self_test() {
    Serial.println("\n===== glucose firmware self-test =====");
    board_info();
    bool pass = glu_shared_beats() != NULL;
    if (!pass) Serial.println("ERROR: could not allocate the beat buffer");
    // 1) device model, calibration-free (running mean over DT_FREE_NW one-minute windows)
    static double avg[DEV_NF], f0[DEV_NF];
    uint32_t best = UINT32_MAX, t_tr = 0;
    double g = NAN;
    for (int rep = 0; rep < 3; rep++) {
        dev_reset();
        uint32_t t0 = 0, t1 = 0;
        for (int w = 0; w < DT_FREE_NW; w++) { t0 = micros(); dev_push_window(DT_FREE_PPG[w], DT_LEN, &g_f, avg); t1 = micros(); }
        g = dev_predict_free(avg, DT_FREE_PROFILE);
        uint32_t t2 = micros();
        if (t2 - t0 < best) { best = t2 - t0; t_tr = t2 - t1; }   // cost of the last (newest) window
    }
    outf("[device model, free] averaging %d window(s) | last window: %d beats, extract %lu us, features %lu us, "
         "trees %lu us, total %lu us | lab %.0f mg/dL\n", dev_avg_windows(), g_f.n_beats, (unsigned long)g_f.us_extract,
         (unsigned long)g_f.us_features, (unsigned long)t_tr, (unsigned long)best, DT_FREE_LABEL);
    pass &= check("free model vs Python", g, DT_FREE_REF, 1e-6);
    // 2) device model, calibrated pair
    dev_reset();
    for (int w = 0; w < DT_CAL0_NW; w++) dev_push_window(DT_CAL0_PPG[w], DT_LEN, &g_f, f0);
    dev_reset();
    for (int w = 0; w < DT_CAL1_NW; w++) dev_push_window(DT_CAL1_PPG[w], DT_LEN, &g_f, avg);
    dev_reset();
    uint32_t t0 = micros();
    double gc = dev_predict_cal(DT_CAL_GLUCOSE, DT_CAL_MINUTES, DT_CAL_PROFILE, avg, f0);
    outf("[device model, calibrated] fingerstick %.0f, %.0f min later -> trees %lu us | lab %.0f mg/dL\n",
         DT_CAL_GLUCOSE, DT_CAL_MINUTES, (unsigned long)(micros() - t0), DT_CAL_LABEL);
#ifdef DEV_FREE_ONLY
    (void)gc;
    Serial.println("  calibrated model not built on this board (DEV_FREE_ONLY)");
#else
    pass &= check("calibrated model vs Python", gc, DT_CAL_REF, 1e-6);
#endif
#ifndef GLU_NO_CNN
    // 3) per-beat CNN (previous approach), timing only + its own reference
    uint32_t t3 = micros();
    glu_estimate(TW_PPG, TW_LEN, 0, &g_res);
    uint32_t cnn = micros() - t3;
    outf("[CNN, for comparison] %d beats, NN %lu us (%.0f us/beat), total %lu us\n", g_res.n_beats,
         (unsigned long)g_res.us_nn, g_res.n_beats ? (double)g_res.us_nn / g_res.n_beats : 0.0, (unsigned long)cnn);
    pass &= check("CNN vs Python", g_res.glucose_mgdl, TW_REF_GLUCOSE, 0.05);
#else
    Serial.println("[CNN] not built on this board (GLU_NO_CNN)");
#endif
    outf("RAM: free %u kB after the test\n", free_ram_kb());
    outf("SELFTEST %s\n", pass ? "PASS" : "FAIL");
    Serial.println("ready: PROFILE age sex_m bmi diabetes | CAL mg/dL | MIN minutes | RESET | BEGIN n ... END | s | BENCH sec");
}

// ---- energy benchmark: read the supply current during each phase -------------------------------
static void idle_until(uint32_t t_end_ms) {
#if defined(ARDUINO_ARCH_ESP32)
    while ((int32_t)(millis() - t_end_ms) < 0) delay(10);          // FreeRTOS idle task: core waits for interrupt
#else
    while ((int32_t)(millis() - t_end_ms) < 0) __WFI();             // sleep until the next 1-ms SysTick
#endif
}

static void bench(int seconds) {
    if (seconds < 5) seconds = 5;
    static double avg[DEV_NF];
    outf("BENCH: %s, %d s IDLE then %d s RUN. Read the current meter in the middle of each phase.\n", BOARD_NAME, seconds, seconds);
    Serial.println("BENCH IDLE start");
    Serial.flush();
    idle_until(millis() + 1000UL * seconds);
    Serial.println("BENCH IDLE end");
    Serial.println("BENCH RUN start");
    Serial.flush();
    uint32_t t_end = millis() + 1000UL * seconds, n = 0, t0 = micros();
    volatile double sink = 0;
    while ((int32_t)(millis() - t_end) < 0) {
        dev_reset();
        dev_push_window(DT_FREE_PPG[DT_FREE_NW - 1], DT_LEN, &g_f, avg);
        sink += dev_predict_free(avg, DT_FREE_PROFILE);
        n++;
    }
    uint32_t us = micros() - t0;
    Serial.println("BENCH RUN end");
    outf("BENCH RESULT board=%s estimates=%lu seconds=%.3f ms_per_estimate=%.3f\n", BOARD_NAME, (unsigned long)n,
         us / 1e6, n ? us / 1e3 / n : 0.0);
    outf("energy per estimate [mJ] = (I_run - I_idle)[mA] x V[V] x %.3f [ms] / 1000\n", n ? us / 1e3 / n : 0.0);
}

void setup() {
#if defined(ARDUINO_ARCH_ESP32)
    Serial.setRxBufferSize(16384);
#endif
    Serial.begin(SERIAL_BAUD);
    for (uint32_t t0 = millis(); !Serial && millis() - t0 < 8000;) delay(10);
    delay(500);
    glu_micros = micros_hook;
    dev_minutes_now = minutes_hook;
    self_test();
}

static char g_line[64];
static int g_len = 0;
void loop() {
    while (Serial.available()) {
        char c = Serial.read();
        if (c == '\n' || c == '\r') {
            if (g_len) {
                g_line[g_len] = 0;
                if (!strcmp(g_line, "s") || !strcmp(g_line, "SELFTEST")) self_test();
                else if (!strncmp(g_line, "BENCH", 5)) bench(atoi(g_line + 5));
                else dev_handle_line(g_line, emit);
                g_len = 0;
            }
        } else if (g_len < (int)sizeof(g_line) - 1) g_line[g_len++] = c;
    }
}

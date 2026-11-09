// Device glucose model: 60-s PPG window -> 15 PPG features -> gradient-boosted trees.
// Mirrors code/device_features.py + 13_device_train.py; verified in esp32/pc_test/device_check.py.
#pragma once
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define DEV_NF 15            // PPG features (order in device_model.h)

typedef struct {
    double f[DEV_NF];        // NAN = not computable
    int    n_beats;
    uint32_t us_extract, us_features;
} dev_features_t;

// features of one 60-s window (raw PPG at 100 Hz)
int dev_features(const float *ppg, int n, dev_features_t *out);

// tree models; inputs exactly as listed in device_model.h
double dev_predict_free(const double *features, const double *profile);
double dev_predict_cal(double ref_glucose, double minutes_since_cal, const double *profile,
                       const double *features, const double *features_at_cal);

// ---- running average over one-minute windows (DEV_AVG_WINDOWS in device_model.h, default 1) ----
void dev_reset(void);                                         // start a new session (forget history)
// features of one window, pushed into the history; avg = NaN-aware mean of the last DEV_AVG_WINDOWS
// windows (windows with < 3 beats count as missing). Returns the number of valid windows in avg.
int  dev_push_window(const float *ppg, int n, dev_features_t *f, double *avg);
int  dev_avg_windows(void);

// ---- command protocol (same code on the ESP32 and in the PC test) ----
//   PROFILE <age> <sex_m 0/1> <bmi> <diabetes 0/1>    set once
//   CAL <glucose mg/dL>                                 next window becomes the calibration
//   MIN <minutes>                                       minutes since calibration for the next window
//                                                       (default on the device: real elapsed time)
//   BEGIN <n> / n sample lines / END                    send one window -> one JSON result line
//                                                       (predictions use the running feature average)
//   RESET                                               new session: clear the feature history
typedef void (*dev_emit_fn)(const char *line);
void dev_handle_line(const char *line, dev_emit_fn emit);
extern double (*dev_minutes_now)(void);  // clock hook for the default MIN; NULL on PC

#ifdef __cplusplus
}
#endif

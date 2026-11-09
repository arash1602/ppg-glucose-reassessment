// PPG -> glucose estimation, plain C (no dynamic memory, no external libraries).
// Mirrors code/ppg.py + models.py TinyCNN exactly; verified against Python in esp32/pc_test.
#pragma once
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define GLU_MAX_SAMPLES 6000    // 60 s at 100 Hz
#define GLU_MAX_PEAKS   400
#ifndef GLU_MAX_BEATS
#define GLU_MAX_BEATS   200     // >= 121: the beat buffer doubles as filter scratch (see glucose.c)
#endif

typedef struct {
    int   n_peaks;              // peaks after distance + prominence selection
    int   n_windows;            // 1-s windows with exactly one peak
    int   n_beats;              // windows passing the template (cosine >= 0.85) check
    float glucose_mgdl;         // median of per-beat predictions (-1 if too few beats)
    float beat_pred[GLU_MAX_BEATS];
    int   beat_peak[GLU_MAX_BEATS];
    uint32_t us_filter, us_beats, us_nn;   // timing breakdown (filled on device)
} glu_result_t;

// accepted 1-s beats of one window (shared by the CNN path and the feature/tree path)
typedef struct {
    int    n_peaks, n_windows, n_beats;
    double accept_frac;          // accepted / windows   (as ppg.extract_beats)
    double cos_mean;             // mean cosine similarity of accepted beats (NAN if none)
    int    peak[GLU_MAX_BEATS];  // sample index of each accepted beat's peak
    float  beat[GLU_MAX_BEATS][100];   // each beat scaled to [-1, 1]
    uint32_t us_filter, us_beats;
} glu_beats_t;

// filter + peak detection + template matching. Returns number of accepted beats, <0 on error.
int glu_extract(const float *ppg, int n, glu_beats_t *out);
glu_beats_t *glu_shared_beats(void);   // one beat buffer shared by all callers; also the filter scratch

// ppg: raw PPG at 100 Hz, n <= GLU_MAX_SAMPLES. use_int8: 1 = int8 weights, 0 = float32 weights.
// Returns 0 on success, <0 on error (-1 too long, -2 too few beats).
int glu_estimate(const float *ppg, int n, int use_int8, glu_result_t *out);

// lower-level pieces (exposed for testing)
void glu_bandpass(const float *x, int n, double *y);                // scipy.signal.sosfiltfilt
int  glu_find_peaks(const double *x, int n, int *peaks);            // find_peaks(distance, prominence)
float glu_nn_beat(const float *beat, int use_int8);                 // one 1-s beat -> mg/dL

// optional timer hook (microseconds); set by the firmware, NULL on PC
extern uint32_t (*glu_micros)(void);

#ifdef __cplusplus
}
#endif

// See glucose.h. Every step reproduces the Python reference (scipy / numpy / PyTorch).
#include "glucose.h"
#ifndef GLU_NO_CNN
#include "glucose_model.h"
#endif
#include "dsp_coeffs.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

uint32_t (*glu_micros)(void) = 0;
static uint32_t now_us(void) { return glu_micros ? glu_micros() : 0; }

/* ------------------------------------------------------------------ */
/* band-pass: scipy.signal.sosfiltfilt(sos, x)  (padtype='odd')        */
/* ------------------------------------------------------------------ */
#define PAD_EDGE (3 * (2 * DSP_N_SOS + 1))   // ntaps*3, no zero coefficients in this design
/* Scratch for the padded filter input and for robust_z: the shared beat buffer, which is only written
   after filtering and peak detection are finished. Saves ~48 kB of RAM (needed on 192-kB parts). */
#define EXT_LEN (GLU_MAX_SAMPLES + 2 * PAD_EDGE)
typedef char glu_scratch_fits[(sizeof(((glu_beats_t *)0)->beat) >= EXT_LEN * sizeof(double)) ? 1 : -1];
#define GLU_SCRATCH() ((double *)glu_shared_beats()->beat)

static void sosfilt_inplace(double *v, int n, double scale_zi) {
    double z[DSP_N_SOS][2];
    for (int s = 0; s < DSP_N_SOS; s++) {
        z[s][0] = DSP_ZI[2 * s] * scale_zi;
        z[s][1] = DSP_ZI[2 * s + 1] * scale_zi;
    }
    for (int i = 0; i < n; i++) {
        double x = v[i];
        for (int s = 0; s < DSP_N_SOS; s++) {
            const double *c = &DSP_SOS[6 * s];          // b0 b1 b2 a0 a1 a2 (a0 = 1)
            double y = c[0] * x + z[s][0];
            z[s][0] = c[1] * x - c[4] * y + z[s][1];
            z[s][1] = c[2] * x - c[5] * y;
            x = y;
        }
        v[i] = x;
    }
}

void glu_bandpass(const float *x, int n, double *y) {
    double *const s_ext = GLU_SCRATCH();
    const int e = PAD_EDGE, m = n + 2 * e;
    for (int i = 0; i < e; i++) s_ext[i] = 2.0 * x[0] - x[e - i];
    for (int i = 0; i < n; i++) s_ext[e + i] = x[i];
    for (int i = 0; i < e; i++) s_ext[e + n + i] = 2.0 * x[n - 1] - x[n - 2 - i];
    sosfilt_inplace(s_ext, m, s_ext[0]);
    for (int i = 0, j = m - 1; i < j; i++, j--) { double t = s_ext[i]; s_ext[i] = s_ext[j]; s_ext[j] = t; }
    sosfilt_inplace(s_ext, m, s_ext[0]);
    for (int i = 0; i < n; i++) y[i] = s_ext[m - 1 - e - i];
}

/* ------------------------------------------------------------------ */
/* robust z-score: (x - median) / (1.4826 * MAD)                        */
/* ------------------------------------------------------------------ */
static int cmp_d(const void *a, const void *b) {
    double x = *(const double *)a, y = *(const double *)b;
    return (x > y) - (x < y);
}
static double sorted_median(double *v, int n) {        // sorts v in place
    qsort(v, n, sizeof(double), cmp_d);
    return (n & 1) ? v[n / 2] : 0.5 * (v[n / 2 - 1] + v[n / 2]);
}
static void robust_z(double *x, int n) {                // scratch is free after filtering
    double *const s_ext = GLU_SCRATCH();
    memcpy(s_ext, x, n * sizeof(double));
    double med = sorted_median(s_ext, n);
    for (int i = 0; i < n; i++) s_ext[i] = fabs(x[i] - med);
    double mad = sorted_median(s_ext, n) * 1.4826;
    if (mad < 1e-8) mad = 1e-8;
    for (int i = 0; i < n; i++) x[i] = (x[i] - med) / mad;
}

/* ------------------------------------------------------------------ */
/* scipy.signal.find_peaks(x, distance=D, prominence=P)                 */
/* order as in scipy: local maxima -> distance -> prominence            */
/* ------------------------------------------------------------------ */
#ifdef GLU_SMALL_RAM
typedef int16_t glu_idx_t;                               // sample indices < 6000 fit in 16 bits (saves 12 kB)
#else
typedef int glu_idx_t;
#endif
static glu_idx_t s_cand[GLU_MAX_SAMPLES / 2];
static glu_idx_t s_order[GLU_MAX_SAMPLES / 2];
static uint8_t s_keep[GLU_MAX_SAMPLES / 2];
static const double *s_sort_x;
static int cmp_prio(const void *a, const void *b) {     // ascending by height, stable by index
    int ia = *(const glu_idx_t *)a, ib = *(const glu_idx_t *)b;
    double x = s_sort_x[s_cand[ia]], y = s_sort_x[s_cand[ib]];
    if (x != y) return (x > y) - (x < y);
    return ia - ib;
}

int glu_find_peaks(const double *x, int n, int *peaks) {
    int m = 0, i = 1, imax = n - 1;
    while (i < imax) {                                   // _local_maxima_1d (plateau aware)
        if (x[i - 1] < x[i]) {
            int ia = i + 1;
            while (ia < imax && x[ia] == x[i]) ia++;
            if (x[ia] < x[i]) { s_cand[m++] = (i + ia - 1) / 2; i = ia; }
        }
        i++;
    }
    for (int k = 0; k < m; k++) { s_keep[k] = 1; s_order[k] = k; }
    s_sort_x = x;
    qsort(s_order, m, sizeof(glu_idx_t), cmp_prio);
    for (int r = m - 1; r >= 0; r--) {                   // _select_by_peak_distance
        int j = s_order[r];
        if (!s_keep[j]) continue;
        for (int k = j - 1; k >= 0 && s_cand[j] - s_cand[k] < DSP_PEAK_DIST; k--) s_keep[k] = 0;
        for (int k = j + 1; k < m && s_cand[k] - s_cand[j] < DSP_PEAK_DIST; k++) s_keep[k] = 0;
    }
    int np = 0;
    for (int k = 0; k < m; k++) {
        if (!s_keep[k]) continue;
        int p = s_cand[k];
        double lmin = x[p], rmin = x[p];                 // _peak_prominences, wlen = whole signal
        for (int a = p; a >= 0 && x[a] <= x[p]; a--) if (x[a] < lmin) lmin = x[a];
        for (int a = p; a < n && x[a] <= x[p]; a++) if (x[a] < rmin) rmin = x[a];
        double prom = x[p] - (lmin > rmin ? lmin : rmin);
        if (prom >= DSP_PEAK_PROM && np < GLU_MAX_PEAKS) peaks[np++] = p;
    }
    return np;
}

#ifndef GLU_NO_CNN
/* ------------------------------------------------------------------ */
/* TinyCNN: 4 x [conv1d(stride 2) + folded BN + ReLU] -> GAP -> FC-ReLU -> FC */
/* ------------------------------------------------------------------ */
static float s_act0[16 * 64], s_act1[16 * 64];

static int conv_relu(const float *in, int len, int layer, int use_int8, float *out) {
    const int ci = GM_SHAPES[layer][0], co = GM_SHAPES[layer][1], k = GM_SHAPES[layer][2];
    const int st = GM_SHAPES[layer][3], pd = GM_SHAPES[layer][4];
    const int olen = (len + 2 * pd - k) / st + 1;
    const float *W = GM_CW[layer], *B = GM_CB[layer];
    const int8_t *WQ = GM_CWQ[layer];
    const float *WS = GM_CWS[layer];
    for (int o = 0; o < co; o++) {
        const float sc = WS[o];
        for (int t = 0; t < olen; t++) {
            float acc = 0.f;
            const int base = t * st - pd;
            for (int c = 0; c < ci; c++) {
                const float *xi = in + c * len;
                const int woff = (o * ci + c) * k;
                for (int j = 0; j < k; j++) {
                    const int idx = base + j;
                    if (idx < 0 || idx >= len) continue;
                    acc += xi[idx] * (use_int8 ? WQ[woff + j] * sc : W[woff + j]);
                }
            }
            acc += B[o];
            out[o * olen + t] = acc > 0.f ? acc : 0.f;
        }
    }
    return olen;
}

float glu_nn_beat(const float *beat, int use_int8) {
    float *a = s_act0, *b = s_act1;
    memcpy(a, beat, GM_IN_LEN * sizeof(float));
    int len = GM_IN_LEN;
    for (int l = 0; l < GM_N_CONV; l++) {
        len = conv_relu(a, len, l, use_int8, b);
        float *t = a; a = b; b = t;
    }
    const int ch = GM_SHAPES[GM_N_CONV - 1][1];
    float gap[64];
    for (int c = 0; c < ch; c++) {
        float s = 0.f;
        for (int t = 0; t < len; t++) s += a[c * len + t];
        gap[c] = s / len;
    }
    float h[GM_FC1_OUT];
    for (int o = 0; o < GM_FC1_OUT; o++) {
        float s = GM_FC0_B[o];
        for (int c = 0; c < ch; c++)
            s += gap[c] * (use_int8 ? GM_FC0_WQ[o * ch + c] * GM_FC0_WS[o] : GM_FC0_W[o * ch + c]);
        h[o] = s > 0.f ? s : 0.f;
    }
    float y = GM_FC1_B[0];
    for (int o = 0; o < GM_FC1_OUT; o++) y += h[o] * (use_int8 ? GM_FC1_WQ[o] * GM_FC1_WS[0] : GM_FC1_W[o]);
    return y * GM_SD + GM_MU;
}

#endif  /* GLU_NO_CNN */

/* ------------------------------------------------------------------ */
/* full pipeline (ppg.extract_beats + per-beat model + median)          */
/* ------------------------------------------------------------------ */
static double s_xz[GLU_MAX_SAMPLES];
static int s_peaks[GLU_MAX_PEAKS];
static int s_win[GLU_MAX_PEAKS];
static double s_tmpl[DSP_BEAT_LEN];
static float s_pred_sorted[GLU_MAX_BEATS];

static int cmp_f(const void *a, const void *b) {
    float x = *(const float *)a, y = *(const float *)b;
    return (x > y) - (x < y);
}

int glu_extract(const float *ppg, int n, glu_beats_t *out) {
    out->n_peaks = out->n_windows = out->n_beats = 0;
    out->accept_frac = 0.0; out->cos_mean = NAN;
    if (n > GLU_MAX_SAMPLES || n < 3 * DSP_BEAT_LEN) return -1;
    uint32_t t0 = now_us();
    glu_bandpass(ppg, n, s_xz);
    robust_z(s_xz, n);
    uint32_t t1 = now_us();
    const int half = DSP_BEAT_LEN / 2;
    int np = glu_find_peaks(s_xz, n, s_peaks), nw = 0;
    out->n_peaks = np;
    for (int i = 0; i < np; i++) {                       // exactly one peak inside [p-50, p+50)
        int p = s_peaks[i], a = p - half, b = p + half;
        if (a < 0 || b > n) continue;
        int cnt = 0;
        for (int j = 0; j < np; j++) cnt += (s_peaks[j] >= a && s_peaks[j] < b);
        if (cnt == 1) s_win[nw++] = a;
    }
    out->n_windows = nw;
    out->us_filter = t1 - t0;
    if (nw < 3) { out->us_beats = now_us() - t1; return 0; }
    for (int k = 0; k < DSP_BEAT_LEN; k++) {             // template = mean window
        double s = 0;
        for (int w = 0; w < nw; w++) s += s_xz[s_win[w] + k];
        s_tmpl[k] = s / nw;
    }
    double tn = 0;
    for (int k = 0; k < DSP_BEAT_LEN; k++) tn += s_tmpl[k] * s_tmpl[k];
    tn = sqrt(tn);
    int nb = 0;
    double cos_sum = 0;
    for (int w = 0; w < nw; w++) {
        const double *x = s_xz + s_win[w];
        double dot = 0, wn = 0, lo = x[0], hi = x[0];
        for (int k = 0; k < DSP_BEAT_LEN; k++) {
            dot += x[k] * s_tmpl[k]; wn += x[k] * x[k];
            if (x[k] < lo) lo = x[k];
            if (x[k] > hi) hi = x[k];
        }
        double cs = dot / (sqrt(wn) * tn + 1e-8);
        if (cs < DSP_COS_MIN || nb >= GLU_MAX_BEATS) continue;
        double rng = hi - lo > 1e-8 ? hi - lo : 1e-8;
        for (int k = 0; k < DSP_BEAT_LEN; k++) out->beat[nb][k] = (float)(2.0 * (x[k] - lo) / rng - 1.0);
        out->peak[nb] = s_win[w] + half;
        cos_sum += cs;
        nb++;
    }
    out->n_beats = nb;
    out->accept_frac = (double)nb / nw;
    out->cos_mean = nb ? cos_sum / nb : NAN;
    out->us_beats = now_us() - t1;
    return nb;
}

static glu_beats_t *s_gbp = 0;
glu_beats_t *glu_shared_beats(void) {           // allocated on first use (heap, or PSRAM if the board has it)
    if (!s_gbp) s_gbp = (glu_beats_t *)malloc(sizeof(glu_beats_t));
    return s_gbp;
}

#ifndef GLU_NO_CNN
int glu_estimate(const float *ppg, int n, int use_int8, glu_result_t *out) {
    memset(out, 0, sizeof(*out));
    out->glucose_mgdl = -1.f;
    glu_beats_t *gb = glu_shared_beats();
    if (!gb) return -3;
#define s_gb (*gb)
    int rc = glu_extract(ppg, n, &s_gb);
    out->n_peaks = s_gb.n_peaks; out->n_windows = s_gb.n_windows; out->n_beats = s_gb.n_beats;
    out->us_filter = s_gb.us_filter; out->us_beats = s_gb.us_beats;
    if (rc < 0) return -1;
    if (s_gb.n_beats < 3) return -2;
    uint32_t nn0 = now_us();
    for (int b = 0; b < s_gb.n_beats; b++) {
        out->beat_pred[b] = glu_nn_beat(s_gb.beat[b], use_int8);
        out->beat_peak[b] = s_gb.peak[b];
    }
    out->us_nn = now_us() - nn0;
    int nb = s_gb.n_beats;
    memcpy(s_pred_sorted, out->beat_pred, nb * sizeof(float));
    qsort(s_pred_sorted, nb, sizeof(float), cmp_f);
    out->glucose_mgdl = (nb & 1) ? s_pred_sorted[nb / 2] : 0.5f * (s_pred_sorted[nb / 2 - 1] + s_pred_sorted[nb / 2]);
    return 0;
}
#undef s_gb
#endif  /* GLU_NO_CNN */

// See device.h. Every number follows code/device_features.py (numpy, float64).
#include "device.h"
#include "device_model.h"
#include "glucose.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define FS 100.0
#define L  100
#define C0 50
#define DEV_ACK_EVERY 250
#ifndef DEV_AVG_WINDOWS
#define DEV_AVG_WINDOWS 1          // older device_model.h: no averaging
#endif
#define DEV_AVG_MAX 16

#define s_b (*glu_shared_beats())
static double s_col[GLU_MAX_BEATS];
static double s_ibi[GLU_MAX_BEATS];
static double s_sq[GLU_MAX_BEATS > 100 ? GLU_MAX_BEATS : 100];   // scratch (kept off the 8 kB loop stack)

extern uint32_t (*glu_micros)(void);
static uint32_t us(void) { return glu_micros ? glu_micros() : 0; }

static int cmp_d(const void *a, const void *b) {
    double x = *(const double *)a, y = *(const double *)b;
    return (x > y) - (x < y);
}
static double median(double *v, int n) {          // sorts v in place (numpy: mean of the 2 middle values)
    qsort(v, n, sizeof(double), cmp_d);
    return (n & 1) ? v[n / 2] : 0.5 * (v[n / 2 - 1] + v[n / 2]);
}
// numpy's pairwise summation (numpy/core/src/umath/loops_utils.h), so sums, means and standard
// deviations match numpy bit for bit (a 1e-16 difference can otherwise flip a tree split)
static double psum(const double *a, int n) {
    if (n < 8) { double r = 0.; for (int i = 0; i < n; i++) r += a[i]; return r; }
    if (n <= 128) {
        double r[8]; int i;
        for (int j = 0; j < 8; j++) r[j] = a[j];
        for (i = 8; i < n - (n % 8); i += 8) for (int j = 0; j < 8; j++) r[j] += a[i + j];
        double res = ((r[0] + r[1]) + (r[2] + r[3])) + ((r[4] + r[5]) + (r[6] + r[7]));
        for (; i < n; i++) res += a[i];
        return res;
    }
    int n2 = n / 2; n2 -= n2 % 8;
    return psum(a, n2) + psum(a + n2, n - n2);
}
static double mean(const double *v, int n) { return psum(v, n) / n; }
static double pstd(const double *v, int n) {      // numpy.std (ddof 0)
    double m = mean(v, n);
    for (int i = 0; i < n; i++) { double t = v[i] - m; s_sq[i] = t * t; }
    return sqrt(psum(s_sq, n) / n);
}
static void gradient(const double *x, double *g, int n, double scale) {   // numpy.gradient * scale
    g[0] = (x[1] - x[0]) * scale;
    for (int i = 1; i < n - 1; i++) g[i] = (x[i + 1] - x[i - 1]) / 2.0 * scale;
    g[n - 1] = (x[n - 1] - x[n - 2]) * scale;
}

enum { HR, IBI_SD, RMSSD, T_RISE, WIDTH50, AREA, MAXS, MINS, APG_BA, APG_EA, DICRO, SKEW, KURT, ACCEPT, COSM };

int dev_features(const float *ppg, int n, dev_features_t *o) {
    for (int i = 0; i < DEV_NF; i++) o->f[i] = NAN;
    if (!glu_shared_beats()) return -3;
    uint32_t t0 = us();
    int rc = glu_extract(ppg, n, &s_b);
    uint32_t t1 = us();
    o->us_extract = t1 - t0;
    o->n_beats = s_b.n_beats;
    if (rc < 0) return rc;
    if (s_b.n_windows >= 3) o->f[ACCEPT] = s_b.accept_frac;
    if (s_b.n_beats) o->f[COSM] = s_b.cos_mean;
    if (s_b.n_beats < 3) { o->us_features = us() - t1; return 0; }

    // inter-beat intervals of accepted beats (0.3 < ibi < 2.0 s)
    int ni = 0;
    for (int i = 1; i < s_b.n_beats; i++) {
        double d = (s_b.peak[i] - s_b.peak[i - 1]) / FS;
        if (d > 0.3 && d < 2.0) s_ibi[ni++] = d;
    }
    if (ni) {
        memcpy(s_col, s_ibi, ni * sizeof(double));
        o->f[HR] = 60.0 / median(s_col, ni);
    }
    if (ni > 2) o->f[IBI_SD] = pstd(s_ibi, ni);
    if (ni > 3) {
        for (int i = 1; i < ni; i++) { double t = s_ibi[i] - s_ibi[i - 1]; s_sq[i - 1] = t * t; }
        o->f[RMSSD] = sqrt(mean(s_sq, ni - 1));
    }

    // template = per-sample median of the accepted beats
    static double T[L], d1[L], d2[L], pos[L], z3[L], z4[L];
    for (int k = 0; k < L; k++) {
        for (int b = 0; b < s_b.n_beats; b++) s_col[b] = (double)s_b.beat[b][k];
        T[k] = median(s_col, s_b.n_beats);
    }
    int foot = 0;
    for (int k = 1; k < C0; k++) if (T[k] < T[foot]) foot = k;
    o->f[T_RISE] = (C0 - foot) / FS;
    double half = (T[C0] + T[foot]) / 2;
    int first = -1, last = -1;
    for (int k = 0; k < L; k++) if (T[k] > half) { if (first < 0) first = k; last = k; }
    if (first >= 0) o->f[WIDTH50] = (last - first) / FS;
    double tmin = T[0];
    for (int k = 1; k < L; k++) if (T[k] < tmin) tmin = T[k];
    for (int k = 0; k < L; k++) pos[k] = T[k] - tmin;
    double s_front = psum(pos, C0), s_all = psum(pos, L);
    o->f[AREA] = s_front / (s_all > 1e-8 ? s_all : 1e-8);
    gradient(T, d1, L, FS);
    double mx = d1[0], mn = d1[0];
    for (int k = 1; k < L; k++) { if (d1[k] > mx) mx = d1[k]; if (d1[k] < mn) mn = d1[k]; }
    o->f[MAXS] = mx; o->f[MINS] = mn;
    gradient(d1, d2, L, FS);
    int ai = 0;
    for (int k = 1; k < C0; k++) if (d2[k] > d2[ai]) ai = k;
    double a = d2[ai], b = d2[ai], e = d2[C0];
    for (int k = ai; k < C0 + 10; k++) if (d2[k] < b) b = d2[k];
    for (int k = C0; k < C0 + 35; k++) if (d2[k] > e) e = d2[k];
    if (a != 0) { o->f[APG_BA] = b / a; o->f[APG_EA] = e / a; }
    double tail = T[C0 + 15];
    for (int k = C0 + 15; k < L; k++) if (T[k] > tail) tail = T[k];
    double den = T[C0] - tmin;
    o->f[DICRO] = (tail - tmin) / (den > 1e-8 ? den : 1e-8);
    double m = mean(T, L), sd = pstd(T, L) + 1e-8;
    for (int k = 0; k < L; k++) { double z = (T[k] - m) / sd; z3[k] = pow(z, 3.0); z4[k] = pow(z, 4.0); }   // numpy z ** 3
    o->f[SKEW] = mean(z3, L); o->f[KURT] = mean(z4, L);
    o->us_features = us() - t1;
    return 0;
}

static double trees(const double *x, int n_trees, const int32_t *root, const int16_t *feat, const double *thr,
                    const int32_t *left, const int32_t *right, const uint8_t *missl, const double *val, double base) {
    double s = base;
    for (int t = 0; t < n_trees; t++) {
        int nd = root[t];
        while (feat[nd] >= 0) {
            double v = x[feat[nd]];
            nd = isnan(v) ? (missl[nd] ? left[nd] : right[nd]) : (v <= thr[nd] ? left[nd] : right[nd]);
        }
        s += val[nd];
    }
    return s;
}

double dev_predict_free(const double *f, const double *profile) {
    double x[GBF_N_INPUTS];
    memcpy(x, f, DEV_NF * sizeof(double));
    memcpy(x + DEV_NF, profile, DEV_N_PROFILE * sizeof(double));
    return trees(x, GBF_N_TREES, gbf_ROOT, gbf_FEAT, gbf_THR, gbf_LEFT, gbf_RIGHT, gbf_MISSL, gbf_VAL, gbf_BASE);
}

#ifdef DEV_FREE_ONLY   /* small-flash boards: the calibrated model is left out */
double dev_predict_cal(double ref, double minutes, const double *profile, const double *f, const double *f0) {
    (void)ref; (void)minutes; (void)profile; (void)f; (void)f0;
    return NAN;
}
#else
double dev_predict_cal(double ref, double minutes, const double *profile, const double *f, const double *f0) {
    double x[GBC_N_INPUTS];
    int k = 0;
    x[k++] = ref; x[k++] = minutes;
    for (int i = 0; i < DEV_N_PROFILE; i++) x[k++] = profile[i];
    for (int i = 0; i < DEV_NF; i++) x[k++] = f[i];
    for (int i = 0; i < DEV_NF; i++) x[k++] = f[i] - f0[i];
    return ref + trees(x, GBC_N_TREES, gbc_ROOT, gbc_FEAT, gbc_THR, gbc_LEFT, gbc_RIGHT, gbc_MISSL, gbc_VAL, gbc_BASE);
}
#endif

/* ------------------------------ protocol ------------------------------ */
double (*dev_minutes_now)(void) = 0;
static float s_buf[GLU_MAX_SAMPLES];
static int s_expect = 0, s_n = 0, s_pending_cal = 0, s_have_cal = 0;
static double s_profile[DEV_N_PROFILE] = {NAN, NAN, NAN, NAN};
static double s_cal_glucose = NAN, s_cal_time = 0, s_minutes_override = NAN;
static double s_cal_f[DEV_NF];
static dev_features_t s_df;

/* running NaN-aware mean of the features of the last DEV_AVG_WINDOWS one-minute windows
   (as 14_device_avg.py: windows with < 3 beats count as all-NaN) */
static double s_hist[DEV_AVG_MAX][DEV_NF];
static int s_hist_n = 0, s_hist_pos = 0;
static double s_avg[DEV_NF];

static void hist_reset(void) { s_hist_n = 0; s_hist_pos = 0; }
static int hist_push_and_mean(const dev_features_t *d, int valid) {   // returns number of valid windows used
    int K = DEV_AVG_WINDOWS < DEV_AVG_MAX ? DEV_AVG_WINDOWS : DEV_AVG_MAX, nvalid = 0;
    for (int i = 0; i < DEV_NF; i++) s_hist[s_hist_pos][i] = valid ? d->f[i] : NAN;
    s_hist_pos = (s_hist_pos + 1) % K;
    if (s_hist_n < K) s_hist_n++;
    for (int w = 0; w < s_hist_n; w++) {
        int any = 0;
        for (int i = 0; i < DEV_NF; i++) if (!isnan(s_hist[w][i])) { any = 1; break; }
        nvalid += any;
    }
    for (int i = 0; i < DEV_NF; i++) {          // numpy order: plain sum in window order, then divide
        double s = 0; int c = 0;
        for (int j = 0; j < s_hist_n; j++) {
            int w = (s_hist_n < K) ? j : (s_hist_pos + j) % K;     // oldest -> newest
            if (!isnan(s_hist[w][i])) { s += s_hist[w][i]; c++; }
        }
        s_avg[i] = c ? s / c : NAN;
    }
    return nvalid;
}

void dev_reset(void) { hist_reset(); }
int dev_avg_windows(void) { return DEV_AVG_WINDOWS; }
int dev_push_window(const float *ppg, int n, dev_features_t *f, double *avg) {
    int rc = dev_features(ppg, n, f);
    int nv = hist_push_and_mean(f, rc == 0 && f->n_beats >= 3);
    memcpy(avg, s_avg, sizeof s_avg);
    return nv;
}

static void fmt_num(char *dst, size_t cap, double v) {
    if (isnan(v)) snprintf(dst, cap, "null"); else snprintf(dst, cap, "%.17g", v);
}

static void run_window(dev_emit_fn emit) {
    static char line[1400];
    char num[32];
    uint32_t t0 = us();
    int rc = dev_features(s_buf, s_n, &s_df);
    uint32_t t1 = us();
    double free_g = NAN, cal_g = NAN, minutes = NAN;
    int n_avg = 0;
    n_avg = hist_push_and_mean(&s_df, rc == 0 && s_df.n_beats >= 3);
    if (rc == 0 && n_avg >= 1) {
        free_g = dev_predict_free(s_avg, s_profile);
        if (s_pending_cal) {
            memcpy(s_cal_f, s_avg, sizeof(s_cal_f));
            s_have_cal = 1; s_pending_cal = 0;
            s_cal_time = dev_minutes_now ? dev_minutes_now() : 0;
            cal_g = s_cal_glucose; minutes = 0;
        } else if (s_have_cal) {
            minutes = !isnan(s_minutes_override) ? s_minutes_override
                      : (dev_minutes_now ? dev_minutes_now() - s_cal_time : NAN);
            cal_g = dev_predict_cal(s_cal_glucose, minutes, s_profile, s_avg, s_cal_f);
        }
    }
    uint32_t t2 = us();
    s_minutes_override = NAN;
    int k = snprintf(line, sizeof line, "{\"tag\":\"window\",\"rc\":%d,\"n\":%d,\"n_beats\":%d,\"n_avg\":%d,\"avg_k\":%d",
                 rc, s_n, s_df.n_beats, n_avg, DEV_AVG_WINDOWS);
    fmt_num(num, sizeof num, free_g); k += snprintf(line + k, sizeof line - k, ",\"glucose_free\":%s", num);
    fmt_num(num, sizeof num, cal_g);  k += snprintf(line + k, sizeof line - k, ",\"glucose_cal\":%s", num);
    fmt_num(num, sizeof num, minutes); k += snprintf(line + k, sizeof line - k, ",\"minutes_since_cal\":%s", num);
    k += snprintf(line + k, sizeof line - k, ",\"us_extract\":%u,\"us_features\":%u,\"us_trees\":%u,\"us_total\":%u,\"features\":[",
                  s_df.us_extract, s_df.us_features, t2 - t1, t2 - t0);
    for (int i = 0; i < DEV_NF; i++) {
        fmt_num(num, sizeof num, s_avg[i]);      // the features the trees used (averaged)
        k += snprintf(line + k, sizeof line - k, "%s%s", i ? "," : "", num);
    }
    snprintf(line + k, sizeof line - k, "]}");
    emit(line);
}

void dev_handle_line(const char *s, dev_emit_fn emit) {
    char msg[160];
    if (!strncmp(s, "BEGIN", 5)) {
        s_expect = atoi(s + 5); s_n = 0;
        if (s_expect <= 0 || s_expect > GLU_MAX_SAMPLES) { emit("{\"error\":\"bad length\"}"); s_expect = 0; }
    } else if (!strncmp(s, "END", 3)) {
        if (s_n > 0) run_window(emit);
        s_expect = 0;
    } else if (!strncmp(s, "PROFILE", 7)) {
        double a, sx, bmi, dm;
        if (sscanf(s + 7, "%lf %lf %lf %lf", &a, &sx, &bmi, &dm) == 4) {
            s_profile[0] = a; s_profile[1] = sx; s_profile[2] = bmi; s_profile[3] = dm;
            snprintf(msg, sizeof msg, "{\"ok\":\"profile\",\"age\":%g,\"sex_m\":%g,\"bmi\":%g,\"diabetes\":%g}", a, sx, bmi, dm);
            emit(msg);
        } else emit("{\"error\":\"PROFILE age sex_m bmi diabetes\"}");
    } else if (!strncmp(s, "CAL", 3)) {
        s_cal_glucose = atof(s + 3); s_pending_cal = 1; s_have_cal = 0;
        snprintf(msg, sizeof msg, "{\"ok\":\"calibration armed\",\"glucose\":%g}", s_cal_glucose);
        emit(msg);
    } else if (!strncmp(s, "RESET", 5)) {       // new session: forget the feature history
        hist_reset();
        emit("{\"ok\":\"reset\"}");
    } else if (!strncmp(s, "MIN", 3)) {
        s_minutes_override = atof(s + 3);
    } else if (s_expect && s_n < s_expect) {
        s_buf[s_n++] = strtof(s, NULL);
        if (s_n % DEV_ACK_EVERY == 0 && s_n < s_expect) {   // flow control: host waits for this
            snprintf(msg, sizeof msg, "{\"ack\":%d}", s_n);
            emit(msg);
        }
    }
}

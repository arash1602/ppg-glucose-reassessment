// PC check: run the firmware C code on test_window.txt and print results as JSON.
#include <stdio.h>
#include "glucose.h"
static float x[GLU_MAX_SAMPLES];
static double xf[GLU_MAX_SAMPLES];
int main(int argc, char **argv) {
    FILE *f = fopen(argc > 1 ? argv[1] : "test_window.txt", "r");
    if (!f) { fprintf(stderr, "no test window\n"); return 1; }
    int n = 0; double v;
    while (n < GLU_MAX_SAMPLES && fscanf(f, "%lf", &v) == 1) x[n++] = (float)v;
    fclose(f);
    glu_bandpass(x, n, xf);
    printf("{\"filtered_first10\":[");
    for (int i = 0; i < 10; i++) printf("%s%.12g", i ? "," : "", xf[i]);
    printf("]");
    for (int q = 0; q < 2; q++) {
        static glu_result_t r;
        int rc = glu_estimate(x, n, q, &r);
        printf(",\"%s\":{\"rc\":%d,\"n_peaks\":%d,\"n_windows\":%d,\"n_beats\":%d,\"glucose_median\":%.6f,\"peaks\":[",
               q ? "int8" : "float", rc, r.n_peaks, r.n_windows, r.n_beats, r.glucose_mgdl);
        for (int i = 0; i < r.n_beats; i++) printf("%s%d", i ? "," : "", r.beat_peak[i]);
        printf("],\"beat_preds\":[");
        for (int i = 0; i < r.n_beats; i++) printf("%s%.6f", i ? "," : "", r.beat_pred[i]);
        printf("]}");
    }
    printf("}\n");
    return 0;
}

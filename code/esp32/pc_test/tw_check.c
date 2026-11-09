#include <stdio.h>
#include <math.h>
#include "glucose.h"
#include "test_window.h"
int main(void) {
    static glu_result_t r;
    for (int q = 0; q < 2; q++) {
        int rc = glu_estimate(TW_PPG, TW_LEN, q, &r);
        printf("%s rc %d beats %d (ref %d) glucose %.4f (Python %.4f) diff %.5f\n", q ? "int8 " : "float", rc, r.n_beats,
               TW_REF_NBEATS, r.glucose_mgdl, TW_REF_GLUCOSE, fabs(r.glucose_mgdl - TW_REF_GLUCOSE));
    }
    return 0;
}

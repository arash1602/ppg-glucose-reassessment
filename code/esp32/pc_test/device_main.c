// PC harness: feeds stdin lines to the same protocol handler the ESP32 uses.
#include <stdio.h>
#include <string.h>
#include "device.h"
static void emit(const char *l) { puts(l); fflush(stdout); }
int main(void) {
    static char line[256];
    while (fgets(line, sizeof line, stdin)) { line[strcspn(line, "\r\n")] = 0; dev_handle_line(line, emit); }
    return 0;
}

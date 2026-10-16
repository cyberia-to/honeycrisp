#include <Accelerate/Accelerate.h>
#include <assert.h>
#include <stdio.h>

_Static_assert(sizeof(vDSP_Stride) == 8, "macOS vDSP signed stride is 64-bit");
_Static_assert(sizeof(vDSP_Length) == 8, "macOS vDSP length is 64-bit");
_Static_assert((vDSP_Stride)-1 < 0, "stride is signed");

int main(void) {
    const float a[] = {1, 2, 3}, b[] = {10, 20, 30};
    float out[3];
    vDSP_vadd(a + 2, -1, b, 1, out, 1, 3);
    assert(out[0] == 13 && out[1] == 22 && out[2] == 31);
    printf("SDK stride=%zu length=%zu; reverse-add={13,22,31}\n",
           sizeof(vDSP_Stride), sizeof(vDSP_Length));
    return 0;
}

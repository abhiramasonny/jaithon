/* builtins_enum.c -- the methods every enum value has, whatever its enum.
 *
 * One today: `ordinal`, the variant's position, which the VM already holds as
 * `ObjEnumVal::tag` and pushes for OP_ENUM_TAG. Exposed because a table
 * indexed by it is 27x cheaper than the `match` chain the front end uses for
 * every enum-to-value lookup (tests/bench/shapes/optable_*.jai), and nothing
 * in the language could read the tag before this.
 *
 * A method the enum itself declares wins: getPropertyInto consults the
 * variant's fields and the enum's own methods first and reaches this table
 * only when both miss. */
#include "runtime/builtins/collections/builtins_seq.h"
#include "runtime/methods.h"
#include "vm/vm.h"

static bool enumValOrdinal(int argc, Value *args, Value *out) {
    (void)argc;
    if (!IS_ENUM_VAL(args[0])) {
        return jaiThrow(vm.cTypeError, "ordinal(): receiver is not an enum value");
    }
    *out = INT_VAL((int64_t)AS_ENUM_VAL(args[0])->tag);
    return true;
}

static const JaiSeqMethod kEnumValMethods[] = {
    JAI_METHOD("ordinal", enumValOrdinal, 1, 1),
};

bool jaiEnumValMethod(Value receiver, ObjString *name, Value *out) {
    return JAI_BIND_FROM(kEnumValMethods, receiver, name, out);
}


import lib/modules/core/wrappers

public static void log in Demo(string label, var value)
    print label + ": " + str(value)
end

public static void main()
    int hardInt = 7
    var dynNum = 3.14
    var dynStr = "hello"
    long big = 100000

    log("hardInt", hardInt)
    log("dynNum", dynNum)
    log("dynStr", dynStr)
    log("big", big)

    string combined = String.concat(Integer.toString(hardInt), "_" + Long.toString(big))
    log("combined", combined)

    dynNum = dynNum + Integer.parseInt("2")
    log("dynNum+2", dynNum)

    dynStr = String.concat(dynStr, "_world")
    log("dynStr+", dynStr)
end

main()

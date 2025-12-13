
import lib/modules/core/wrappers

public static void main()
    int count = 5
    long big = 12345678901
    float f = 1.5
    double d = 2.25
    short s = 12
    byte b = 7
    char c = "Z"
    bool flag = true

    print "== Raw primitives =="
    print count
    print big
    print f
    print d
    print s
    print b
    print c
    print flag

    print "== Boxed conversions =="
    print Integer.toString(count)
    print Long.toString(big)
    print Float.toString(f)
    print Double.toString(d)
    print Char.toString(c)
    print Bool.toString(flag)
    print String.valueOf(count + big)

    print "== Parsing =="
    print Integer.parseInt("42")
    print Long.parseLong("9001")
    print Float.parseFloat("3.5")
    print Double.parseDouble("6.25")
    print Bool.parseBool("true")
end

main()

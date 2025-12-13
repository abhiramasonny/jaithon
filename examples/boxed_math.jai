
import lib/modules/core/wrappers

public static void main()
    int a = 10
    double b = 2.5
    long c = 1000
    float d = 3.0

    print "sum=" + str(a + b + c + d)

    int fromStr = Integer.parseInt("15")
    long big = Long.valueOf(c + fromStr)
    string bigStr = Long.toString(big)
    print "big=" + bigStr

    float fParsed = Float.parseFloat("4.5")
    double mix = Double.valueOf(fParsed) + Double.valueOf(1.5)
    print "mix=" + Double.toString(mix)

    char letter = Char.valueOf("X")
    print "char=" + Char.toString(letter)

    bool truth = Bool.parseBool("true")
    print "bool=" + Bool.toString(truth)
end

main()

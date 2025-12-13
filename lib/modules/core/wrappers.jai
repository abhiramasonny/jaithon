public int valueOf in Integer(int x)
    return x
end

public int parseInt in Integer(string s)
    return num(s)
end

public string toString in Integer(int x)
    return str(x)
end

public long valueOf in Long(long x)
    return x
end

public long parseLong in Long(string s)
    return num(s)
end

public string toString in Long(long x)
    return str(x)
end

public short valueOf in Short(short x)
    return x
end

public short parseShort in Short(string s)
    return num(s)
end

public string toString in Short(short x)
    return str(x)
end

public byte valueOf in Byte(byte x)
    return x
end

public byte parseByte in Byte(string s)
    return num(s)
end

public string toString in Byte(byte x)
    return str(x)
end

public float valueOf in Float(float x)
    return x
end

public float parseFloat in Float(string s)
    return num(s)
end

public string toString in Float(float x)
    return str(x)
end

public double valueOf in Double(double x)
    return x
end

public double parseDouble in Double(string s)
    return num(s)
end

public string toString in Double(double x)
    return str(x)
end

public char valueOf in Char(char c)
    return c
end

public string toString in Char(char c)
    return str(c)
end

public bool valueOf in Bool(bool b)
    return b
end

public bool parseBool in Bool(string s)
    return s == "true"
end

public string toString in Bool(bool b)
    if b
        return "true"
    end
    return "false"
end

public string valueOf in String(var any)
    return str(any)
end

public int length in String(string s)
    return len(s)
end

public string concat in String(string a, string b)
    return a + b
end

public char charAt in String(string s, int idx)
    return charAt(s, idx)
end

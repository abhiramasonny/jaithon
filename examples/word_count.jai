import ds/hashmap
import core/string

public bool isAlpha(string c)
    return indexOf("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ", c) >= 0
end

public string normalize(string text)
    string lowerText = lower(text)
    string out = ""
    int i = 0
    while i < len(lowerText)
        string ch = charAt(lowerText, i)
        if isAlpha(ch) or ch == " "
            out = concat(out, ch)
        else
            out = concat(out, " ")
        end
        i = i + 1
    end
    return trim(out)
end

public var splitWords(string text)
    var words = []
    int start = 0
    int i = 0
    while i < len(text)
        if charAt(text, i) == " "
            if i > start
                _push(words, substr(text, start, i - start))
            end
            start = i + 1
        end
        i = i + 1
    end
    if start < len(text)
        _push(words, substr(text, start, len(text) - start))
    end
    return words
end

public var countWords(var words)
    var counts = hashmap()
    int i = 0
    while i < _alen(words)
        var w = _get(words, i)
        var cur = counts.get(w)
        if cur == null
            counts.set(w, 1)
        else
            counts.set(w, cur + 1)
        end
        i = i + 1
    end
    return counts
end

public void showCounts(var counts)
    var keys = counts.keys()
    int i = 0
    while i < _alen(keys)
        var k = _get(keys, i)
        var v = counts.get(k)
        print k + ": " + str(v)
        i = i + 1
    end
end

public static void main()
    string text = "Test! This is a sample text, with: punctuation. This text is for testing, testing word-count. Word-count should count words correctly; words, words, words. Now let's see how well it works? Does it count words correctly?"
    string clean = normalize(text)
    var words = splitWords(clean)
    var counts = countWords(words)
    showCounts(counts)
end

main()

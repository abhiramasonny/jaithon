#frequency counter using hashmap

import lib/modules/hashmap
import lib/modules/string

func isAlpha(c)
    return (c >= "a" and c <= "z") or (c >= "A" and c <= "Z")
end

func normalize(text)
    var out = ""
    var i = 0
    while i < len(text)
        var ch = charAt(text, i)
        if ch >= "A" and ch <= "Z"
            var upper = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
            var idx = indexOf(upper, ch)
            if idx >= 0
                ch = charAt("abcdefghijklmnopqrstuvwxyz", idx)
            end
        end
        if isAlpha(ch) or ch == " "
            out = concat(out, ch)
        else
            out = concat(out, " ")
        end
        i = i + 1
    end
    return trim(out)
end

func splitWords(text)
    var words = []
    var start = 0
    var i = 0
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

func countWords(words)
    var counts = hashmap()
    var i = 0
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

func showCounts(counts)
    var keys = counts.keys()
    var i = 0
    while i < _alen(keys)
        var k = _get(keys, i)
        var v = counts.get(k)
        print k + ": " + str(v)
        i = i + 1
    end
end

func main()
    var text = "To be, or not to be, that is the question: Whether 'tis nobler in the mind to suffer The slings and arrows of outrageous fortune"
    var clean = normalize(text)
    var words = splitWords(clean)
    var counts = countWords(words)
    showCounts(counts)
end

main()

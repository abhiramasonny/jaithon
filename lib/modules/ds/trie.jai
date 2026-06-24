class TrieNode
    var children
    var isEnd
    var count

    func init(self)
        self.children = _dict()
        self.isEnd = false
        self.count = 0
    end
end

class Trie
    var root
    var size

    func init(self)
        self.root = new TrieNode()
        self.size = 0
    end

    func insert(self, word)
        var node = self.root
        var i = 0
        while i < len(word)
            var c = charAt(word, i)
            if not _dict_has(node.children, c)
                node.children[c] = new TrieNode()
            end
            node = node.children[c]
            node.count = node.count + 1
            i = i + 1
        end
        if not node.isEnd
            node.isEnd = true
            self.size = self.size + 1
        end
    end

    func search(self, word)
        var node = self.root
        var i = 0
        while i < len(word)
            var c = charAt(word, i)
            if not _dict_has(node.children, c)
                return false
            end
            node = node.children[c]
            i = i + 1
        end
        return node.isEnd
    end

    func startsWith(self, prefix)
        var node = self.root
        var i = 0
        while i < len(prefix)
            var c = charAt(prefix, i)
            if not _dict_has(node.children, c)
                return false
            end
            node = node.children[c]
            i = i + 1
        end
        return true
    end

    func countPrefix(self, prefix)
        var node = self.root
        var i = 0
        while i < len(prefix)
            var c = charAt(prefix, i)
            if not _dict_has(node.children, c)
                return 0
            end
            node = node.children[c]
            i = i + 1
        end
        return node.count
    end

    func _collect(self, node, prefix, results)
        if node.isEnd
            _push(results, prefix)
        end
        var keys = _dict_keys(node.children)
        var i = 0
        while i < _alen(keys)
            self._collect(node.children[keys[i]], concat(prefix, keys[i]), results)
            i = i + 1
        end
    end

    func wordsWithPrefix(self, prefix)
        var node = self.root
        var i = 0
        while i < len(prefix)
            var c = charAt(prefix, i)
            if not _dict_has(node.children, c)
                return []
            end
            node = node.children[c]
            i = i + 1
        end
        var results = []
        self._collect(node, prefix, results)
        return results
    end

    func allWords(self)
        return self.wordsWithPrefix("")
    end

    func delete(self, word)
        if not self.search(word)
            return false
        end
        var node = self.root
        var i = 0
        while i < len(word)
            var c = charAt(word, i)
            node = node.children[c]
            node.count = node.count - 1
            i = i + 1
        end
        node.isEnd = false
        self.size = self.size - 1
        return true
    end
end

func trieFrom(words)
    var t = new Trie()
    var i = 0
    while i < _alen(words)
        t.insert(words[i])
        i = i + 1
    end
    return t
end

func longestIncreasingSubsequence(arr)
    var n = _alen(arr)
    if n == 0
        return 0
    end
    var dp = []
    var i = 0
    while i < n
        _push(dp, 1)
        i = i + 1
    end
    var best = 1
    i = 1
    while i < n
        var j = 0
        while j < i
            if arr[j] < arr[i] and dp[j] + 1 > dp[i]
                dp[i] = dp[j] + 1
            end
            j = j + 1
        end
        if dp[i] > best
            best = dp[i]
        end
        i = i + 1
    end
    return best
end

func knapsack01(weights, values, capacity)
    var n = _alen(weights)
    var dp = []
    var i = 0
    while i <= capacity
        _push(dp, 0)
        i = i + 1
    end
    i = 0
    while i < n
        var j = capacity
        while j >= weights[i]
            var withItem = dp[j - weights[i]] + values[i]
            if withItem > dp[j]
                dp[j] = withItem
            end
            j = j - 1
        end
        i = i + 1
    end
    return dp[capacity]
end

func coinChange(coins, amount)
    var dp = []
    var i = 0
    while i <= amount
        _push(dp, amount + 1)
        i = i + 1
    end
    dp[0] = 0
    i = 1
    while i <= amount
        var j = 0
        while j < _alen(coins)
            if coins[j] <= i
                var candidate = dp[i - coins[j]] + 1
                if candidate < dp[i]
                    dp[i] = candidate
                end
            end
            j = j + 1
        end
        i = i + 1
    end
    if dp[amount] > amount
        return -1
    end
    return dp[amount]
end

func coinChangeWays(coins, amount)
    var dp = []
    var i = 0
    while i <= amount
        _push(dp, 0)
        i = i + 1
    end
    dp[0] = 1
    i = 0
    while i < _alen(coins)
        var j = coins[i]
        while j <= amount
            dp[j] = dp[j] + dp[j - coins[i]]
            j = j + 1
        end
        i = i + 1
    end
    return dp[amount]
end

func maxSubarraySum(arr)
    if _alen(arr) == 0
        return 0
    end
    var best = arr[0]
    var current = arr[0]
    var i = 1
    while i < _alen(arr)
        if current + arr[i] > arr[i]
            current = current + arr[i]
        else
            current = arr[i]
        end
        if current > best
            best = current
        end
        i = i + 1
    end
    return best
end

func maxSubarrayProduct(arr)
    if _alen(arr) == 0
        return 0
    end
    var best = arr[0]
    var maxP = arr[0]
    var minP = arr[0]
    var i = 1
    while i < _alen(arr)
        var v = arr[i]
        var newMax = max(v, max(maxP * v, minP * v))
        var newMin = min(v, min(maxP * v, minP * v))
        maxP = newMax
        minP = newMin
        if maxP > best
            best = maxP
        end
        i = i + 1
    end
    return best
end

func climbStairs(n)
    if n <= 1
        return 1
    end
    var a = 1
    var b = 1
    var i = 2
    while i <= n
        var temp = a + b
        a = b
        b = temp
        i = i + 1
    end
    return b
end

func minPathSum(grid)
    var rows = _alen(grid)
    var cols = _alen(grid[0])
    var dp = []
    var i = 0
    while i < rows
        var row = []
        var j = 0
        while j < cols
            _push(row, 0)
            j = j + 1
        end
        _push(dp, row)
        i = i + 1
    end
    dp[0][0] = grid[0][0]
    i = 1
    while i < rows
        dp[i][0] = dp[i - 1][0] + grid[i][0]
        i = i + 1
    end
    var j = 1
    while j < cols
        dp[0][j] = dp[0][j - 1] + grid[0][j]
        j = j + 1
    end
    i = 1
    while i < rows
        j = 1
        while j < cols
            if dp[i - 1][j] < dp[i][j - 1]
                dp[i][j] = dp[i - 1][j] + grid[i][j]
            else
                dp[i][j] = dp[i][j - 1] + grid[i][j]
            end
            j = j + 1
        end
        i = i + 1
    end
    return dp[rows - 1][cols - 1]
end

func wordBreak(s, wordDict)
    var n = len(s)
    var dp = []
    var i = 0
    while i <= n
        _push(dp, false)
        i = i + 1
    end
    dp[0] = true
    i = 1
    while i <= n
        var j = 0
        while j < i
            if dp[j]
                var word = substr(s, j, i - j)
                if arrayContains(wordDict, word)
                    dp[i] = true
                end
            end
            j = j + 1
        end
        i = i + 1
    end
    return dp[n]
end

func longestIncreasingSubsequenceSeq(arr)
    var n = _alen(arr)
    if n == 0
        return []
    end
    var dp = []
    var prev = []
    var i = 0
    while i < n
        _push(dp, 1)
        _push(prev, -1)
        i = i + 1
    end
    var bestLen = 1
    var bestEnd = 0
    i = 1
    while i < n
        var j = 0
        while j < i
            if arr[j] < arr[i] and dp[j] + 1 > dp[i]
                dp[i] = dp[j] + 1
                prev[i] = j
            end
            j = j + 1
        end
        if dp[i] > bestLen
            bestLen = dp[i]
            bestEnd = i
        end
        i = i + 1
    end
    var result = []
    var idx = bestEnd
    while idx >= 0
        _push(result, arr[idx])
        idx = prev[idx]
    end
    return arrayReverse(result)
end

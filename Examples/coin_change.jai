#dynamic programming

import lib/modules/core

func countWays(amount, coins)
    var dp = []
    var i = 0
    while i <= amount
        _push(dp, 0)
        i = i + 1
    end
    _set(dp, 0, 1)
    
    var c = 0
    while c < _alen(coins)
        var coin = _get(coins, c)
        var a = coin
        while a <= amount
            var existing = _get(dp, a)
            var add = _get(dp, a - coin)
            _set(dp, a, existing + add)
            a = a + 1
        end
        c = c + 1
    end
    
    return _get(dp, amount)
end

func main()
    var coins = [1, 2, 5]
    var amount = 10
    print "Coins: [1, 2, 5]"
    print "Amount: " + str(amount)
    print "Ways to make change: " + str(countWays(amount, coins))
    
    coins = [3, 4, 7]
    amount = 15
    print ""
    print "Coins: [3, 4, 7]"
    print "Amount: " + str(amount)
    print "Ways to make change: " + str(countWays(amount, coins))
end

main()


import lib/modules/ds/array

func main()
    print "Testing arrayPush with array..."
    var matrix = []
    var row = [1, 2]
    arrayPush(matrix, row)
    print "Matrix len: " + str(arrayLen(matrix))
    print "Matrix[0] type: " + str(matrix[0])
    print "Matrix[0][0]: " + str(matrix[0][0])
end

main()

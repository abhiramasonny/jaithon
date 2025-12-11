#IO Module

func open(path, mode)
    return _fopen(path, mode)
end

func close(file)
    return _fclose(file)
end

func read(file)
    return _fread(file)
end

func write(file, content)
    return _fwrite(file, content)
end

func readFile(path)
    var f = open(path, "r")
    if f == null
        return null
    end
    var content = read(f)
    close(f)
    return content
end

func writeFile(path, content)
    var f = open(path, "w")
    if f == null
        return false
    end
    write(f, content)
    close(f)
    return true
end

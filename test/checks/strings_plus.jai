# String helpers and casts
# Expected output:
# ababab
# 1
# 1
# 1
# CAMEL
# mixed
# one three three
# ait
# Jaithon
# 6
# s

print repeat("ab", 3)
print startsWith("jaithon", "jai")
print endsWith("jaithon", "thon")
print contains_str("bootstrapped language", "strap")
print upper("Camel")
print lower("MiXed")
print replace("one two two", "two", "three")
print substr("jaithon", 1, 3)
print concat("Jai", "thon")
print len("shapes")
print charAt("snakes", 0)

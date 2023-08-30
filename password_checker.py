# password_checker
# scipt checks if the password has ever been hacked

import requests
import hashlib

def request_api_data(query_char):
    url = "https://api.pwnedpasswords.com/range/" + query_char
    res = requests.get(url)
    if res.status_code != 200:
        raise RuntimeError(
            f"Error fetching: {res.status_code}, check api and try again"
        )
    return res

def get_password_leaks_count(hashes, hash_to_check):
    hashes = (line.split(":") for line in hashes.text.splitlines())
    for h, count in hashes:
        if h == hash_to_check:
            return count
    return 0

def pwned_api_check(password):
    sha1_password = hashlib.sha1(password.encode("utf-8")).hexdigest().upper()
    first5_char, tail = sha1_password[:5], sha1_password[5:]
    response = request_api_data(first5_char)
    count = get_password_leaks_count(response, tail)
    
    suggestions = []
    if count:
        suggestions.append("Change your password.")
        if len(password) < 8:
            suggestions.append("Consider using a password with at least 8 characters.")
        if not any(c.islower() for c in password):
            suggestions.append("Include lowercase letters in your password.")
        if not any(c.isupper() for c in password):
            suggestions.append("Include uppercase letters in your password.")
        if not any(c.isdigit() for c in password):
            suggestions.append("Include numbers in your password.")
        if not any(c in "!@#$%^&*()_-+=<>?/" for c in password):
            suggestions.append("Include symbols in your password.")

    return count, suggestions

# if __name__ == "__main__":
#     sys.exit(main(sys.argv[1:]))

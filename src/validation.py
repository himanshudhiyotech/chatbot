# Validate 

import re

class validation:
    def validatePhoneNumber(phone):
        phoneRegex = r'^(?:\+91|)[1-9][0-9]{9}$'
        if re.match(phoneRegex, phone):
            return True
        return False

    def validateEmail(email):
        emailRegex = r'^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+$'
        if re.match(emailRegex, email):
            return True
        return False
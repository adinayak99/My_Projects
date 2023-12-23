def convert_date(date_string):
    # Remove the timezone offset from the string
    date_string_without_tz = date_string[:-4]

    # Convert the string to datetime object
    date_object = datetime.strptime(date_string_without_tz, "%Y/%m/%d %H:%M:%S")

    # Convert the datetime object to desired date format
    formatted_date = date_object.strftime("%m-%d-%Y")
    
    return formatted_date
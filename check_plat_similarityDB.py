import pyodbc
import textdistance

def find_similar_plat(ocr_plat):
    driver_db = "ODBC Driver 17 for SQL Server"
    server_db = "xxx.xxx.xxx.xxx"
    dbname = "xxxxx"
    username = "xxx"
    password = "xxxxx"
    timeout = 1
    connection_string = f'DRIVER={driver_db};SERVER={server_db};DATABASE={dbname};UID={username};PWD={password};Connect Timeout={timeout}'
    
    # Connect to the database
    conn = pyodbc.connect(connection_string)
    cursor = conn.cursor()
    
    # Query to fetch all [number_plat] values
    query = "SELECT [number_plat] FROM [xxxx].[dbo].[VEHICLEMASTER]"
    cursor.execute(query)
    
    best_similarity = 0
    most_similar_plat = None
    
    # Compare OCR result with each [number_plat] and find the most similar one
    for row in cursor.fetchall():
        db_plat = row[0]
        similarity = textdistance.jaro_winkler(ocr_plat, db_plat)
        
        if similarity > best_similarity:
            best_similarity = similarity
            most_similar_plat = db_plat
    
    conn.close()
    
    return most_similar_plat, best_similarity

# Provide the OCR result
ocr_plat = 'L966E'
most_similar_plat, similarity_score = find_similar_plat(ocr_plat)

print(f"Most similar plat: {most_similar_plat}")
print(f"Similarity score: {similarity_score}")

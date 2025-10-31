# db.py
import pymysql

def dbconnection():
    connection = pymysql.connect(
        host='localhost',
        database='traffic_system',
        user='root',
        password='H_b_1726'
    )
    return connection

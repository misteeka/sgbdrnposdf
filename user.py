import psycopg2
from psycopg2.extras import NamedTupleCursor
from db import get_conn

def init_table():
    conn = get_conn()
    with conn.cursor() as curs:
        curs.execute("create table users (username VARCHAR(50) PRIMARY KEY, password VARCHAR(36) NOT NULL);")
    conn.commit()
    conn.close()
    
def login(username, password):
    #conn = get_conn()
    #realPass = ''
    #with conn.cursor(cursor_factory=NamedTupleCursor) as curs:
    #    curs.execute("SELECT password FROM users WHERE username=%s", (username,))
    #    realPass = curs.fetchone()[0]
    #conn.close()
    #return password == realPass
    return True

def register(username, password):
    #conn = get_conn()
    #with conn.cursor(cursor_factory=NamedTupleCursor) as curs:
    #    curs.execute("INSERT INTO users (username, password) VALUES (%s, %s)", (username, password))
    #conn.commit()
    #conn.close()
    return True
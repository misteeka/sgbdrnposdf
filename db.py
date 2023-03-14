import psycopg2
from psycopg2.extras import NamedTupleCursor

def get_conn():
    try:
        conn = psycopg2.connect('postgresql://postgres:emalak@127.0.0.1:41092/nto')
        return conn
    except:
        # в случае сбоя подключения будет выведено сообщение в STDOUT
        print('Can`t establish connection to database')
        return
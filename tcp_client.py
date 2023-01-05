import socket

target_host = "192.168.1.254"
target_port = 80
client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
client.connect((target_host, target_port))
client.send(b"GET / HTTP/1.1\r\nHost: 192.168.1.254\r\n\r\n")
response = client.recv(4096)
print(response.decode())
client.close()
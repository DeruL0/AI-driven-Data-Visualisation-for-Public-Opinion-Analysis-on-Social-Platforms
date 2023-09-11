# 创建本地服务器并打开项目网页
import webbrowser
import socket
from http.server import SimpleHTTPRequestHandler
from socketserver import TCPServer


class CORSRequestHandler(SimpleHTTPRequestHandler):
    def end_headers(self):
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        super().end_headers()


def check_port_available(port):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM)as s:
        return s.connect_ex(('localhost', port)) != 0


def main():
    port = 8080

    while not check_port_available(port):
        port += 1

    Handler = CORSRequestHandler
    httpd = TCPServer(("", port), Handler)

    print(f"Serving on port {port}")
    webbrowser.open(f"http://localhost:{port}/index.html")  # 联系html文件

    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass

    httpd.server_close()
    httpd.socket.close()  # 关闭套接字以释放端口
    print(f"Stopped serving on port {port}")


if __name__ == "__main__":
    main()

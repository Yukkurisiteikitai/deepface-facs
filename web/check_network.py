"""
ネットワーク接続診断スクリプト
"""
import socket
import subprocess
import sys

def get_local_ips():
    """すべてのローカルIPを取得"""
    ips = []
    try:
        # 方法1: socket
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ips.append(("socket", s.getsockname()[0]))
        s.close()
    except:
        pass
    
    try:
        # 方法2: hostname
        hostname = socket.gethostname()
        ip = socket.gethostbyname(hostname)
        if ip not in [i[1] for i in ips]:
            ips.append(("hostname", ip))
    except:
        pass
    
    return ips

def check_port(port):
    """ポートが使用可能か確認"""
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.settimeout(1)
    try:
        sock.bind(("0.0.0.0", port))
        sock.close()
        return True, "利用可能"
    except socket.error as e:
        return False, str(e)

def check_listening(port):
    """ポートがリッスン中か確認"""
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.settimeout(1)
    try:
        result = sock.connect_ex(("127.0.0.1", port))
        sock.close()
        return result == 0
    except:
        return False

def main():
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8000
    
    print("=" * 50)
    print("🔍 ネットワーク診断")
    print("=" * 50)
    
    # IPアドレス
    print("\n📍 ローカルIPアドレス:")
    ips = get_local_ips()
    for method, ip in ips:
        print(f"   {ip} ({method})")
    
    if ips:
        main_ip = ips[0][1]
        print(f"\n📱 スマホからのアクセスURL:")
        print(f"   http://{main_ip}:{port}")
    
    # ポート状態
    print(f"\n🔌 ポート {port} の状態:")
    if check_listening(port):
        print(f"   ✅ ポート {port} はリッスン中")
    else:
        available, msg = check_port(port)
        if available:
            print(f"   ⚠️ ポート {port} は空いているがサーバーが起動していない")
        else:
            print(f"   ❌ ポート {port} は使用中: {msg}")
    
    # ファイアウォール確認（macOS）
    print("\n🛡️ ファイアウォール:")
    try:
        result = subprocess.run(
            ["/usr/libexec/ApplicationFirewall/socketfilterfw", "--getglobalstate"],
            capture_output=True, text=True
        )
        if "enabled" in result.stdout.lower():
            print("   ⚠️ ファイアウォールが有効です")
            print("   → システム環境設定 > セキュリティとプライバシー > ファイアウォール")
            print("   → Pythonの接続を許可するか、一時的に無効化してください")
        else:
            print("   ✅ ファイアウォールは無効")
    except:
        print("   ❓ ファイアウォール状態を確認できません")
    
    # 同じネットワーク確認
    print("\n📡 確認事項:")
    print("   1. スマホとPCが同じWiFiに接続されているか")
    print("   2. ルーターのAP分離（クライアント分離）が無効か")
    print("   3. VPNが有効になっていないか")
    
    print("\n" + "=" * 50)

if __name__ == "__main__":
    main()

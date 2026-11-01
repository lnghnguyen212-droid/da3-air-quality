import requests

API_LATEST = "https://beijing-observation-simulator.onrender.com/api/v1/live/latest"
API_LIVE = "https://beijing-observation-simulator.onrender.com/api/v1/live"

def get_latest_observation():
    """Lấy bản ghi quan trắc mới nhất từ dịch vụ realtime"""
    try:
        response = requests.get(API_LATEST, timeout=15)
        if response.status_code == 200:
            return response.json()
        print(f"Lỗi gọi API: {response.status_code}")
        return None
    except Exception as e:
        print(f"Không thể kết nối tới dịch vụ quan trắc: {e}")
        return None

if __name__ == "__main__":
    data = get_latest_observation()
    print("Dữ liệu quan trắc nhận được:", data)
import os, io, cv2, torch, numpy as np, base64, tempfile
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from PIL import Image
import torchvision.transforms as transforms
import torch.nn as nn
import torchvision.models as models
import uuid
from flask import send_file


app = Flask(__name__, static_folder='static', static_url_path='/static')
#CORS(app)
CORS(app, resources={r"/api/*": {"origins": "*"}}, supports_credentials=True)

# ── CONFIG ──────────────────────────────────────────────────────────
MODEL_PATH = "best_depth_model_v8.pth"  
IMG_SIZE   = 256
DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── V8 ARCHITECTURE ─────────────────────────────────────────────────
class DecoderBlock(nn.Module):
    def __init__(self, in_ch, skip_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch + skip_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
        )
    def forward(self, x, skip=None):
        x = nn.functional.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        if skip is not None:
            x = torch.cat([x, skip], dim=1)
        return self.conv(x)

class ResNet50UNet(nn.Module):
    def __init__(self):
        super().__init__()
        base = models.resnet50(weights=None)
        self.enc0 = nn.Sequential(base.conv1, base.bn1, base.relu)  # 64ch  H/2
        self.pool  = base.maxpool
        self.enc1  = base.layer1   # 256ch  H/4
        self.enc2  = base.layer2   # 512ch  H/8
        self.enc3  = base.layer3   # 1024ch H/16
        self.enc4  = base.layer4   # 2048ch H/32

        # dec4 input: 2048 (from enc4) + 1024 (skip enc3) = 3072  ← key fix
        self.dec4 = DecoderBlock(2048, 1024, 512)
        # dec3 input: 512 + 512 (skip enc2) = 1024
        self.dec3 = DecoderBlock(512,  512,  256)
        # dec2 input: 256 + 256 (skip enc1) = 512
        self.dec2 = DecoderBlock(256,  256,  128)
        # dec1 input: 128 + 64 (skip enc0) = 192
        self.dec1 = DecoderBlock(128,  64,   64)

        # final_upsample = Conv(64→32, 3×3) + BN  (then bilinear 2×)
        self.final_upsample = nn.Sequential(
            nn.Conv2d(64, 32, 3, padding=1),
            nn.BatchNorm2d(32),
        )

        # out_conv = Conv(32→1, 1×1) + Sigmoid
        self.out_conv = nn.Sequential(
            nn.Conv2d(32, 1, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        e0 = self.enc0(x)
        e1 = self.enc1(self.pool(e0))
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)
        d  = self.dec4(e4, e3)
        d  = self.dec3(d,  e2)
        d  = self.dec2(d,  e1)
        d  = self.dec1(d,  e0)
        d  = nn.functional.interpolate(d, scale_factor=2, mode='bilinear', align_corners=False)
        d  = self.final_upsample(d)
        d  = nn.functional.interpolate(d, scale_factor=2, mode='bilinear', align_corners=False)
        return self.out_conv(d)
    

# ── LOAD MODEL ──────────────────────────────────────────────────────
model = ResNet50UNet().to(DEVICE)
if os.path.exists(MODEL_PATH):
    state = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False)
    clean = {k.replace("module.", ""): v for k, v in state.items()}  # strip DataParallel prefix
    model.load_state_dict(clean)
    print(f"[OK] Model loaded from {MODEL_PATH} on {DEVICE}")
else:
    print(f"[WARN] No checkpoint at {MODEL_PATH} — using random weights")
model.eval()

# ── TRANSFORMS ──────────────────────────────────────────────────────
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

# ── INFERENCE HELPER ────────────────────────────────────────────────
def predict(pil_img):
    """PIL RGB image → plasma-colourised depth PNG bytes (same size as input)."""
    inp = transform(pil_img).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        depth = model(inp).squeeze().cpu().numpy()          # [H,W]  0-1
    #depth_u8  = (depth * 255).astype(np.uint8)
    depth_u8 = (255 - depth * 255).astype(np.uint8)
    coloured  = cv2.applyColorMap(depth_u8, cv2.COLORMAP_PLASMA)
    coloured  = cv2.resize(coloured, pil_img.size)          # back to original size
    _, buf    = cv2.imencode('.png', coloured)
    return buf.tobytes()

def to_b64(data: bytes) -> str:
    return base64.b64encode(data).decode()

def pil_to_b64(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format='PNG')
    return to_b64(buf.getvalue())

# ── ROUTES ──────────────────────────────────────────────────────────

@app.route('/')
def index():
    return send_from_directory('static', 'index.html')

# 1. Single image
@app.route('/api/image', methods=['POST'])
def api_image():
    if 'file' not in request.files:
        return jsonify(error="No file"), 400
    img = Image.open(request.files['file'].stream).convert('RGB')
    return jsonify(original=pil_to_b64(img), depth=to_b64(predict(img)))

# 2. Single webcam frame (base64 JPEG from browser)
@app.route('/api/frame', methods=['POST'])
def api_frame():
    data = request.get_json()
    if not data or 'frame' not in data:
        return jsonify(error="No frame"), 400
    raw = base64.b64decode(data['frame'].split(',')[-1])
    img = Image.open(io.BytesIO(raw)).convert('RGB')
    return jsonify(depth="data:image/png;base64," + to_b64(predict(img)))

# 3. Video file
@app.route('/api/video', methods=['POST'])
def api_video():
    if 'file' not in request.files:
        return jsonify(error="No file"), 400

    f = request.files['file']
    uid = str(uuid.uuid4())[:8]

    with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as tmp:
        f.save(tmp.name)
        tmp_path = tmp.name

    out_path = os.path.join(app.static_folder, f'depth_{uid}.mp4')

    import imageio
    cap = cv2.VideoCapture(tmp_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    W   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    writer = imageio.get_writer(out_path, fps=fps, codec='libx264', quality=7)
    n = 0
    while True: #n < 50:
        ret, frame = cap.read()
        if not ret: break
        pil     = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        dep     = cv2.imdecode(np.frombuffer(predict(pil), np.uint8), cv2.IMREAD_COLOR)
        dep     = cv2.resize(dep, (W, H))
        dep_rgb = cv2.cvtColor(dep, cv2.COLOR_BGR2RGB)
        frm_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        writer.append_data(np.hstack([frm_rgb, dep_rgb]))
        n += 1

    cap.release()
    writer.close()
    os.unlink(tmp_path)

    # Send the file directly — no static URL needed
    response = send_file(
        out_path,
        mimetype='video/mp4',
        as_attachment=False,
        download_name=f'depth_{uid}.mp4'
    )
    response.headers['Content-Disposition'] = f'inline; filename=depth_{uid}.mp4'

    # Clean up after sending
    @response.call_on_close
    def cleanup():
        try: os.unlink(out_path)
        except: pass

    return response


@app.after_request
def after_request(response):
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type')
    response.headers.add('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
    return response


if __name__ == '__main__':
    print("Open http://localhost:8080 in your browser")
    app.run(debug=True, port=8080)
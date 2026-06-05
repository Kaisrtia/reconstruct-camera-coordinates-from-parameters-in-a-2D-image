# Camera PnP Reconstruction

Ung dung web Python dung de tai tao phep chieu phoi canh nguoc tu anh 2D. He thong giai bai toan Perspective-n-Point (PnP), uoc luong pose camera, tinh sai so chieu lai, va ve truc toa do 3D len anh goc de chung minh ket qua anh xa 2D/3D.

## Muc Tieu De Tai

Bai toan can giai:

```text
Object Points 3D + Image Points 2D + Intrinsic Matrix K
=> solvePnP
=> Rotation R, Translation t
=> Camera Center C_world
=> Reprojection Error
=> 3D axis overlay on original image
```

Ung dung hien tai ho tro anh checkerboard. Tu so goc trong cua checkerboard va kich thuoc o vuong, chuong trinh tu sinh toa do 3D thuc te tren mat phang `Z = 0`, detect toa do 2D tren anh, roi dung OpenCV de giai PnP.

## Cong Nghe

- Python
- Streamlit cho dashboard UI
- OpenCV cho checkerboard detection, `solvePnP`, `projectPoints`, ve overlay
- NumPy cho tinh toan ma tran
- Pandas cho bang du lieu diem 2D/3D
- Matplotlib cho demo CLI cu

## Ly Thuyet Cot Loi

Mo hinh chieu pinhole camera:

```text
s [u, v, 1]^T = K [R | t] [X, Y, Z, 1]^T
```

Trong do:

- `K`: ma tran noi tai camera.
- `R`: ma tran xoay tu world space sang camera space.
- `t`: vector tinh tien trong bien doi world-to-camera.
- `[X, Y, Z]`: diem 3D trong world/object space.
- `[u, v]`: diem 2D tren anh.
- `s`: he so ti le homogeneous.

OpenCV dung quy uoc:

```text
X_camera = R * X_world + t
```

Vi vay `t` khong phai truc tiep la vi tri camera. Toa do tam camera trong world space la:

```text
C_world = -R^T * t
```

Sai so danh gia duoc tinh bang reprojection error:

```text
error_i = || image_point_i - reprojected_point_i ||
RMSE = sqrt(mean(error_i^2))
```

## Cau Truc Du An

```text
src/camera_reconstruction/
  core/
    camera_model.py       Mo hinh K, R, t, P va camera center
    projection.py         Chieu 3D world points sang 2D image points
    reconstruction.py     Back-project pixel 2D thanh ray/diem camera khi co depth
    checkerboard.py       Detect checkerboard va sinh object points 3D
    pnp_solver.py         solvePnP, Rodrigues, reprojection error
    overlay.py            Ve truc X/Y/Z len anh goc

  services/
    reconstruction_service.py
      Pipeline xu ly: image + params -> PnP result + overlay image

  storage/
    files.py              Luu upload va output runtime

  web/
    streamlit_app.py      Dashboard UI chinh
    main.py               FastAPI UI cu, khong phai entrypoint chinh
    routes.py
    templates/
    static/

scripts/
  run_web.py              Chay dashboard Streamlit
  run_demo.py             Chay CLI demo cu

tests/
  test_smoke.py
  test_pnp_solver.py
```

Ba file wrapper `camera_model.py`, `projection.py`, `reconstruction.py` o cap package van duoc giu de cac import cu va demo CLI khong bi gay.

## Cai Dat

Tren Windows PowerShell:

```powershell
cd D:\Dev\reconstruct-camera-coordinates-from-parameters-in-a-2D-image
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e .
```

Neu da cai moi truong roi, chi can:

```powershell
python -m pip install -e .
```

## Chay Ung Dung Web

```powershell
python scripts\run_web.py
```

Sau do mo:

```text
http://127.0.0.1:8501
```

Co the doi port bang bien moi truong:

```powershell
$env:CAMERA_RECONSTRUCTION_PORT="8502"
python scripts\run_web.py
```

## Cach Su Dung Dashboard

1. Mo ung dung Streamlit.
2. Trong sidebar, upload anh checkerboard.
3. Nhap `Corners X` va `Corners Y`.
   - Day la so goc trong cua checkerboard, khong phai so o vuong.
   - Vi du ban co 10 o ngang va 7 o doc thi so goc trong thuong la `9 x 6`.
4. Nhap `Square size`.
   - Co the dung `1.0` neu chi can toa do theo don vi o vuong.
   - Neu biet kich thuoc that, vi du 2.5 cm, nhap `2.5`.
5. Mo expander `Intrinsic Matrix K` neu muon nhap camera calibration that.
   - Neu khong biet `K`, de mac dinh khong dung custom K.
6. Mo expander `Distortion` neu co distortion coefficients.
7. Bam `Run PnP`.

## Ket Qua Hien Thi

Dashboard hien thi:

- Anh goc da duoc ve truc toa do 3D.
- RMSE, Mean Error, Max Error bang `st.metric`.
- So diem checkerboard detect duoc.
- `K`, `R`, `P`, `t`, `rvec`, `C_world` bang cong thuc LaTeX.
- Bang doi chieu:
  - Image 2D `[u, v]`
  - World 3D `[X, Y, Z]`
  - Camera 3D `[Xc, Yc, Zc]`
  - Reprojected 2D
  - Error px

## Giai Thich Luong Xu Ly

```text
Upload image
  -> OpenCV doc anh
  -> Detect checkerboard corners 2D
  -> Sinh object points 3D tren mat phang Z = 0
  -> Tao hoac nhan ma tran K
  -> cv2.solvePnP(object_points, image_points, K, dist)
  -> Lay rvec, tvec
  -> cv2.Rodrigues(rvec) thanh R
  -> Tinh C_world = -R^T t
  -> cv2.projectPoints de chieu lai diem 3D
  -> Tinh RMSE/Mean/Max reprojection error
  -> Project truc 3D X/Y/Z len anh
  -> Render dashboard
```

## Chay CLI Demo

```powershell
python scripts\run_demo.py
```

Demo CLI in cac ma tran va luu anh minh hoa vao thu muc `outputs/`.

## Chay Test

```powershell
python -m unittest discover -s tests
```

## Thu Muc Runtime

Anh upload va anh overlay duoc luu trong:

```text
var/uploads/
var/results/
```

Thu muc `var/` da duoc dua vao `.gitignore`, nen khong bi commit vao repository.

## Luu Y Khi Viet Bao Cao

- Day la bai toan PnP, khong phai khoi phuc toan bo canh 3D tu mot anh don le.
- Tu mot pixel 2D rieng le, khong the suy ra duy nhat mot diem 3D neu khong co depth hoac rang buoc hinh hoc.
- Trong bai nay, rang buoc hinh hoc den tu checkerboard: cac diem 3D thuc te tren mat phang `Z = 0` da duoc biet truoc.
- Ket qua can bao cao gom `R`, `tvec`, `C_world`, reprojection error va anh overlay truc 3D.
- Neu `K` khong dung voi camera that, pose uoc luong van co the fit diem anh nhung y nghia vat ly se kem chinh xac hon.


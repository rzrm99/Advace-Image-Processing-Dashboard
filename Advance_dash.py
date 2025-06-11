

# imagedash_final_fixed.py   • 2025-06-13
# ---------------------------------------------------------------------------
# pip install PyQt6 pyqtgraph opencv-python matplotlib numpy
# python imagedash_final_fixed.py
# ---------------------------------------------------------------------------
import sys, cv2, numpy as np
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel, QPushButton, QFileDialog,
    QComboBox, QVBoxLayout, QHBoxLayout, QGridLayout, QFormLayout,
    QSlider, QLineEdit, QSplitter, QStackedWidget, QTabWidget
)
from PyQt6.QtCore import Qt, QEasingCurve, QPropertyAnimation, QPointF, pyqtSignal
from PyQt6.QtGui import QFont, QPixmap, QImage
import pyqtgraph as pg
from PyQt6.QtCore import QEvent  # Make sure this is at the top of your file
# ---------------- helper ----------------------------------------------------
def cv2qt(img):
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w, ch = rgb.shape
    return QPixmap.fromImage(QImage(rgb.data, w, h, ch*w, QImage.Format.Format_RGB888))

def add_noise(img, kind):
    r,c,ch = img.shape
    if kind=="gaussian":
        return np.clip(img+np.random.normal(0,15,(r,c,ch)),0,255).astype(np.uint8)
    if kind=="salt_pepper":
        out,amt,s_vs_p = img.copy(),.02,.5
        n=int(amt*img.size*s_vs_p); coords=[np.random.randint(0,i-1,n) for i in img.shape]
        out[tuple(coords)]=255
        n=int(amt*img.size*(1-s_vs_p)); coords=[np.random.randint(0,i-1,n) for i in img.shape]
        out[tuple(coords)]=0; return out
    if kind=="poisson":
        vals=2**np.ceil(np.log2(len(np.unique(img))))
        return np.clip(np.random.poisson(img*vals)/vals,0,255).astype(np.uint8)
    return img

# ---------------- ImageEditor ----------------------------------------------
class ImageEditor(QWidget):
    image_updated = pyqtSignal(np.ndarray)
    def __init__(self):
        super().__init__()
        self.img_orig=None; self.img_proc=None
        # preview
        self.lbl_img=QLabel(alignment=Qt.AlignmentFlag.AlignCenter)
        self.lbl_img.setMinimumSize(512,384); self.lbl_img.setStyleSheet("border:1px solid black;")
        # controls
        self.btn_load=QPushButton("Load Image",clicked=self.load_image)
        self.btn_save=QPushButton("Save Result",clicked=self.save_image)
        self.btn_reset=QPushButton("Reset",clicked=self.reset_editor)
        self.cmb_color=QComboBox(); self.cmb_color.addItems(["Original","Grayscale","RGB"])
        self.cmb_noise=QComboBox(); self.cmb_noise.addItems(["None","Gaussian","Salt & Pepper","Poisson"])
        self.cmb_filter=QComboBox(); self.cmb_filter.addItems(["None","Blur","GaussianBlur","Median","Bilateral","Edge Detection","Sharpen","Custom"])
        for cb in (self.cmb_color,self.cmb_noise,self.cmb_filter): cb.currentIndexChanged.connect(self.apply_pipeline)
        self.kernel=[[QLineEdit("0") for _ in range(3)] for _ in range(3)]; kgrid=QGridLayout()
        for i in range(3):
            for j in range(3):
                self.kernel[i][j].setFixedWidth(40); kgrid.addWidget(self.kernel[i][j],i,j)
        self.btn_run_kernel=QPushButton("Run Custom Filter",clicked=self.apply_custom_kernel)
        self.cmb_example=QComboBox(); self.cmb_example.addItems(["Select Example…","Sharpen","Edge Detection","Emboss"])
        self.cmb_example.currentIndexChanged.connect(self.insert_kernel_example)
        self.sld_bri=QSlider(Qt.Orientation.Horizontal,minimum=-100,maximum=100,value=0)
        self.sld_con=QSlider(Qt.Orientation.Horizontal,minimum=-100,maximum=100,value=0)
        self.lbl_bri,self.lbl_con=QLabel("0"),QLabel("0")
        self.sld_bri.valueChanged.connect(self.update_bri_con); self.sld_con.valueChanged.connect(self.update_bri_con)
        def row(*w):
            c=QWidget(); h=QHBoxLayout(c); h.setContentsMargins(0,0,0,0); [h.addWidget(x) for x in w]; return c
        form=QFormLayout()
        form.addRow(self.btn_load); form.addRow("Color Mode:",self.cmb_color); form.addRow("Noise:",self.cmb_noise)
        form.addRow("Filter:",self.cmb_filter); form.addRow("Custom Kernel:",kgrid); form.addRow(self.btn_run_kernel)
        form.addRow("Insert Example:",self.cmb_example); form.addRow(self.btn_reset)
        form.addRow("Brightness:",row(self.sld_bri,self.lbl_bri)); form.addRow("Contrast:",row(self.sld_con,self.lbl_con)); form.addRow(self.btn_save)
        left=QWidget(); left.setLayout(form)
        split=QSplitter(Qt.Orientation.Horizontal); split.addWidget(left); split.addWidget(self.lbl_img); split.setStretchFactor(1,3)
        QVBoxLayout(self).addWidget(split)
    # tone curve hook
    def apply_lut(self,lut):
        if self.img_proc is None: return
        self.img_proc=cv2.LUT(self.img_proc,lut); self.update_view()
    # slots
    def load_image(self):
        fp,_=QFileDialog.getOpenFileName(self,"Open Image")
        if not fp: return
        self.img_orig=cv2.imread(fp); self.img_proc=self.img_orig.copy()
        self.reset_sliders(); self.update_view()
    def save_image(self):
        if self.img_proc is None: return
        fp,_=QFileDialog.getSaveFileName(self,"Save","","PNG (*.png);;JPEG (*.jpg)")
        if fp: cv2.imwrite(fp,self.img_proc)
    def reset_editor(self):
        if self.img_orig is None: return
        self.img_proc=self.img_orig.copy()
        self.cmb_color.setCurrentIndex(0); self.cmb_noise.setCurrentIndex(0); self.cmb_filter.setCurrentIndex(0); self.cmb_example.setCurrentIndex(0)
        for r in self.kernel: [c.setText("0") for c in r]
        self.reset_sliders(); self.update_view()
    def reset_sliders(self):
        for s in (self.sld_bri,self.sld_con): s.blockSignals(True); s.setValue(0); s.blockSignals(False)
        self.lbl_bri.setText("0"); self.lbl_con.setText("0")
    def apply_pipeline(self):
        if self.img_orig is None: return
        img=self.img_orig.copy()
        if self.cmb_color.currentText()=="Grayscale":
            img=cv2.cvtColor(cv2.cvtColor(img,cv2.COLOR_BGR2GRAY),cv2.COLOR_GRAY2BGR)
        elif self.cmb_color.currentText()=="RGB":
            img=cv2.cvtColor(cv2.cvtColor(img,cv2.COLOR_BGR2RGB),cv2.COLOR_RGB2BGR)
        nz=self.cmb_noise.currentText().lower().replace(" ","_")
        if nz!="none": img=add_noise(img,nz)
        f=self.cmb_filter.currentText()
        if f=="Blur": img=cv2.blur(img,(3,3))
        elif f=="GaussianBlur": img=cv2.GaussianBlur(img,(5,5),0)
        elif f=="Median": img=cv2.medianBlur(img,5)
        elif f=="Bilateral": img=cv2.bilateralFilter(img,9,75,75)
        elif f=="Edge Detection": img=cv2.filter2D(img,-1,np.array([[-1,-1,-1],[-1,8,-1],[-1,-1,-1]]))
        elif f=="Sharpen": img=cv2.filter2D(img,-1,np.array([[0,-1,0],[-1,5,-1],[0,-1,0]]))
        self.img_proc=img; self.reset_sliders(); self.update_view()
    def apply_custom_kernel(self):
        if self.img_proc is None: return
        try:
            k=np.array([[float(self.kernel[i][j].text()) for j in range(3)] for i in range(3)])
            self.img_proc=cv2.filter2D(self.img_proc,-1,k); self.update_view()
        except ValueError: print("bad kernel")
    def insert_kernel_example(self):
        ex={"Sharpen":[[0,-1,0],[-1,5,-1],[0,-1,0]],
            "Edge Detection":[[-1,-1,-1],[-1,8,-1],[-1,-1,-1]],
            "Emboss":[[-2,-1,0],[-1,1,1],[0,1,2]]}
        name=self.cmb_example.currentText()
        if name in ex:
            for i in range(3):
                for j in range(3):
                    self.kernel[i][j].setText(str(ex[name][i][j]))
    def update_bri_con(self):
        if self.img_proc is None: return
        bri,con=self.sld_bri.value(),self.sld_con.value(); self.lbl_bri.setText(str(bri)); self.lbl_con.setText(str(con))
        img=self.img_proc.astype(np.float32); img=img*(con/50+1)-con+bri
        self.update_view(np.clip(img,0,255).astype(np.uint8))
    def update_view(self,override=None):
        if self.img_proc is None: return
        show=override if override is not None else self.img_proc
        self.lbl_img.setPixmap(cv2qt(show).scaled(self.lbl_img.size(),Qt.AspectRatioMode.KeepAspectRatio,Qt.TransformationMode.SmoothTransformation))
        self.image_updated.emit(show)

# ---------------- Charts ----------------------------------------------------
class ChartsPage(QWidget):
    def __init__(self,editor):
        super().__init__()
        editor.image_updated.connect(self.refresh)
        self.hist=pg.PlotWidget(background='k',title="RGB Histogram"); self.hist.showGrid(x=True,y=True,alpha=0.3)
        self.x=np.arange(256); self.curves=[self.hist.plot(pen=c) for c in ('r','g','b')]
        self.fft=pg.ImageView(); self.fft.ui.histogram.hide(); self.fft.ui.roiBtn.hide(); self.fft.ui.menuBtn.hide()
        tabs=QTabWidget(); tabs.addTab(self.hist,"Histogram"); tabs.addTab(self.fft,"FFT Spectrum")
        QVBoxLayout(self).addWidget(tabs)
    def refresh(self,img):
        [c.setData(self.x,cv2.calcHist([img],[i],None,[256],[0,256]).flatten()) for i,c in enumerate(self.curves)]
        gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY); mag=20*np.log(np.abs(np.fft.fftshift(np.fft.fft2(gray)))+1)
        self.fft.setImage(mag,autoLevels=True,autoRange=False)

# ---------------- Curves page (event-filter version) ----------------------
class CurvesPage(QWidget):
    """Interactive tone-curve editor with four draggable anchors."""
    def __init__(self, editor: ImageEditor):
        super().__init__()
        self.editor = editor

        # ── plot ---------------------------------------------------------
        self.plot = pg.PlotWidget(background='k')
        self.plot.setXRange(0, 255); self.plot.setYRange(0, 255)
        self.plot.showGrid(x=True, y=True, alpha=0.3)
        self.plot.setMouseEnabled(x=False, y=False)

        self.hist_curve  = pg.PlotCurveItem(pen=pg.mkPen((80, 80, 80, 120)))
        self.tone_curve  = pg.PlotCurveItem(pen='c')
        self.plot.addItem(self.hist_curve)
        self.plot.addItem(self.tone_curve)

        # four default anchors
        self.points = [
            QPointF(0,   0),
            QPointF(85,  85),
            QPointF(170, 170),
            QPointF(255, 255)
        ]
        self.scatter = pg.ScatterPlotItem(
            size=8, symbol='o', brush='w', pen=pg.mkPen('k')
        )
        self.plot.addItem(self.scatter)

        # ── right column -------------------------------------------------
        self.preview = QLabel(alignment=Qt.AlignmentFlag.AlignCenter)
        self.preview.setMinimumSize(300, 225)
        self.preview.setStyleSheet("border:1px solid gray;")

        btn_reset = QPushButton("Reset Curve", clicked=self._reset_curve)

        rhs = QVBoxLayout(); rhs.addWidget(self.preview); rhs.addWidget(btn_reset)
        rhs_w = QWidget(); rhs_w.setLayout(rhs)

        splitter = QSplitter(); splitter.addWidget(self.plot); splitter.addWidget(rhs_w)
        splitter.setStretchFactor(0, 3)
        QVBoxLayout(self).addWidget(splitter)

        # keep preview / hist in sync with ImageEditor
        editor.image_updated.connect(self._update_hist)
        editor.image_updated.connect(self._update_preview)

        # install one event filter for press / move / release handling
        self._drag_index = None
        self.plot.scene().installEventFilter(self)

        self._redraw()          # first draw

    # ─────────────── QWidget override ────────────────────────────────────
    def eventFilter(self, obj, ev):
        if obj is not self.plot.scene():
            return super().eventFilter(obj, ev)

        t = ev.type()
        if t == QEvent.Type.GraphicsSceneMousePress:
            if ev.button() == Qt.MouseButton.LeftButton:
                self._start_drag(ev)
            return False  # allow other items to see the event
        elif t == QEvent.Type.GraphicsSceneMouseMove:
            self._drag_move(ev)
            return False
        elif t == QEvent.Type.GraphicsSceneMouseRelease:
            self._drag_index = None
            return False

        return super().eventFilter(obj, ev)

    # ─────────────── internal helpers ───────────────────────────────────
    def _redraw(self):
        xs = [p.x() for p in self.points]
        ys = [p.y() for p in self.points]
        self.scatter.setData(pos=np.column_stack((xs, ys)))   # easiest form
        self.tone_curve.setData(xs, ys)

        lut = np.interp(np.arange(256), xs, ys).astype(np.uint8)
        self.editor.apply_lut(lut)

    def _reset_curve(self):
        # Reset the anchor points
        self.points = [
            QPointF(0, 0),
            QPointF(85, 85),
            QPointF(170, 170),
            QPointF(255, 255)
        ]

        # Reset the image in the editor to original
        if self.editor.img_orig is not None:
            self.editor.img_proc = self.editor.img_orig.copy()
            self.editor.update_view()

        # Reapply identity LUT and refresh visuals
        self._redraw()
        if self.editor.img_proc is not None:
            self._update_hist(self.editor.img_proc)
            self._update_preview(self.editor.img_proc)

    # ─────────────── drag handling ──────────────────────────────────────
    def _nearest_anchor(self, pos_view, tol=8):
        for i, p in enumerate(self.points):
            if abs(p.x() - pos_view.x()) < tol and abs(p.y() - pos_view.y()) < tol:
                return i
        return None

    def _start_drag(self, ev):
        pos_view = self.plot.getViewBox().mapSceneToView(ev.scenePos())
        self._drag_index = self._nearest_anchor(pos_view)

    def _drag_move(self, ev):
        if self._drag_index is None:
            return
        if not QApplication.mouseButtons() & Qt.MouseButton.LeftButton:
            return

        view = self.plot.getViewBox().mapSceneToView(ev.scenePos())
        x = int(np.clip(view.x(), 0, 255))
        y = int(np.clip(view.y(), 0, 255))

        # maintain strict X order
        if self._drag_index > 0:
            x = max(x, self.points[self._drag_index - 1].x() + 1)
        if self._drag_index < len(self.points) - 1:
            x = min(x, self.points[self._drag_index + 1].x() - 1)

        self.points[self._drag_index] = QPointF(x, y)
        self._redraw()

    # ─────────────── histogram & preview ────────────────────────────────
    def _update_hist(self, img):
        if img is None:
            self.hist_curve.clear(); return
        h = cv2.calcHist([img], [0], None, [256], [0, 256]).flatten()
        if h.max() == 0: return
        self.hist_curve.setData(np.arange(256), h / h.max() * 255)

    def _update_preview(self, img):
        if img is None:
            self.preview.clear(); return
        self.preview.setPixmap(
            cv2qt(img).scaled(
                self.preview.size(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation
            )
        )

# ---------------- Morphology (kernel label fixed) ---------------------------
class MorphologyPage(QWidget):
    def __init__(self,editor):
        super().__init__()
        self.editor=editor; self.snapshot=None; editor.image_updated.connect(self.set_preview)
        self.lbl_prev=QLabel(alignment=Qt.AlignmentFlag.AlignCenter); self.lbl_prev.setMinimumSize(400,300); self.lbl_prev.setStyleSheet("border:1px solid gray;")
        self.sld=QSlider(Qt.Orientation.Horizontal,minimum=1,maximum=25,value=3,singleStep=2,pageStep=2)
        self.lbl_sz=QLabel("3"); self.sld.valueChanged.connect(lambda v:self.lbl_sz.setText(str(v if v%2 else v+1)))
        self.ops=[("Dilate",cv2.dilate),("Erode",cv2.erode),("Open",lambda i,k:cv2.morphologyEx(i,cv2.MORPH_OPEN,k)),
                  ("Close",lambda i,k:cv2.morphologyEx(i,cv2.MORPH_CLOSE,k)),("Gradient",lambda i,k:cv2.morphologyEx(i,cv2.MORPH_GRADIENT,k)),
                  ("Top Hat",lambda i,k:cv2.morphologyEx(i,cv2.MORPH_TOPHAT,k)),("Black Hat",lambda i,k:cv2.morphologyEx(i,cv2.MORPH_BLACKHAT,k))]
        grid=QGridLayout(); [grid.addWidget(QPushButton(n,clicked=lambda _,name=n:self.apply(name)),*divmod(i,3)) for i,(n,_) in enumerate(self.ops)]
        self.btn_reset=QPushButton("Reset Morphology",clicked=self.reset)
        def row(*w): c=QWidget(); h=QHBoxLayout(c); h.setContentsMargins(0,0,0,0); [h.addWidget(x) for x in w]; return c
        form=QFormLayout(); form.addRow("Kernel size:",row(self.sld,self.lbl_sz)); form.addRow(grid); form.addRow(self.btn_reset)
        left=QWidget(); left.setLayout(form)
        split=QSplitter(Qt.Orientation.Horizontal); split.addWidget(left); split.addWidget(self.lbl_prev); split.setStretchFactor(1,2)
        QVBoxLayout(self).addWidget(split)
    def take_snap(self):
        if self.editor.img_proc is not None: self.snapshot=self.editor.img_proc.copy()
    def apply(self,name):
        if self.editor.img_proc is None: return
        if self.snapshot is None: self.take_snap()
        k=self.sld.value(); k+=k%2==0
        kernel=cv2.getStructuringElement(cv2.MORPH_RECT,(k,k))
        res=dict(self.ops)[name](self.editor.img_proc,kernel)
        self.editor.img_proc=res; self.editor.update_view(); self.set_preview(res)
    def reset(self):
        if self.snapshot is None: return
        self.editor.img_proc=self.snapshot.copy(); self.editor.update_view(); self.set_preview(self.snapshot); self.sld.setValue(3)
    def set_preview(self,img):
        if img is None: self.lbl_prev.clear(); return
        self.lbl_prev.setPixmap(cv2qt(img).scaled(self.lbl_prev.size(),Qt.AspectRatioMode.KeepAspectRatio,Qt.TransformationMode.SmoothTransformation))

# ---------------- Simple pages ---------------------------------------------
class HomePage(QWidget):
    def __init__(self,e):
        super().__init__(); lbl=QLabel("🏠 Welcome!\n\nUse the sidebar to load an image and start experimenting.",alignment=Qt.AlignmentFlag.AlignCenter); lbl.setFont(QFont("Arial",16)); btn=QPushButton("Load an image now",clicked=e.load_image); lay=QVBoxLayout(self); lay.addStretch(); lay.addWidget(lbl); lay.addWidget(btn,alignment=Qt.AlignmentFlag.AlignCenter); lay.addStretch()
class SettingsPage(QWidget):
    def __init__(self,main):
        super().__init__(); cbo=QComboBox(); cbo.addItems(["Light","Dark"]); cbo.setCurrentIndex(1); cbo.currentIndexChanged.connect(lambda i:main.set_theme(dark=i==1)); lay=QVBoxLayout(self); lay.addWidget(QLabel("⚙️ Theme")); lay.addWidget(cbo); lay.addStretch()

# ---------------- Main window ----------------------------------------------
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("🧠 Image-Processing Dashboard"); self.resize(1150,720)
        self.editor=ImageEditor(); self.charts=ChartsPage(self.editor); self.curves=CurvesPage(self.editor); self.morph=MorphologyPage(self.editor)
        # sidebar
        self.sidebar=QWidget(); sbl=QVBoxLayout(self.sidebar)
        nav=[("🏠 Home",0),("🖼 Editor",1),("📊 Charts",2),("🎚 Curves",3),("🧩 Morphology",4),("⚙️ Settings",5)]
        for t,i in nav:
            b=QPushButton(t,clicked=lambda _,idx=i:self.pages.setCurrentIndex(idx)); b.setMinimumHeight(40); b.setFont(QFont("Arial",12)); b.setObjectName("navButton"); sbl.addWidget(b)
        self.btn_coll=QPushButton("⇄ Collapse",clicked=self.toggle_sidebar); self.btn_coll.setObjectName("navButton"); sbl.addWidget(self.btn_coll); sbl.addStretch()
        # pages
        self.pages=QStackedWidget(); self.pages.addWidget(HomePage(self.editor)); self.pages.addWidget(self.editor); self.pages.addWidget(self.charts); self.pages.addWidget(self.curves); self.pages.addWidget(self.morph); self.pages.addWidget(SettingsPage(self))
        split=QSplitter(Qt.Orientation.Horizontal); split.addWidget(self.sidebar); split.addWidget(self.pages); split.setSizes([200,900])
        QVBoxLayout(cont:=QWidget()).addWidget(split); self.setCentralWidget(cont)
        self.set_theme(True)
    def toggle_sidebar(self):
        w=self.sidebar.width(); tgt=50 if w>100 else 200
        anim=QPropertyAnimation(self.sidebar,b"maximumWidth",duration=300); anim.setStartValue(w); anim.setEndValue(tgt); anim.setEasingCurve(QEasingCurve.Type.InOutQuad); anim.start(); self._hold=anim
    def set_theme(self,dark=True):
        if dark: ss="QMainWindow{background:#1f1f2f;} QWidget{background:#1f1f2f;color:#fff;} QPushButton#navButton{background:#2f2f3f;border:none;padding:10px;text-align:left;} QPushButton#navButton:hover{background:#3f3f5f;} QPushButton{background:#44446a;border:1px solid #626289;border-radius:5px;padding:6px 10px;} QPushButton:hover{background:#57578a;}"
        else:    ss="QMainWindow{background:#f4f4f4;} QWidget{background:#f4f4f4;color:#000;} QPushButton#navButton{background:#ddd;border:none;padding:10px;text-align:left;} QPushButton#navButton:hover{background:#ccc;} QPushButton{background:#eee;border:1px solid #bababa;border-radius:5px;padding:6px 10px;} QPushButton:hover{background:#dcdcdc;}"
        self.setStyleSheet(ss)

# ---------------- run -------------------------------------------------------
if __name__=="__main__":
    app=QApplication(sys.argv)
    MainWindow().show(); sys.exit(app.exec())
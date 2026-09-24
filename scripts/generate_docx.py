"""生成项目说明文档 Word 版本"""

from docx import Document
from docx.shared import Pt, Inches, Cm, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.oxml.ns import qn
import os


def set_cell_shading(cell, color):
    """设置单元格背景色"""
    shading = cell._element.get_or_add_tcPr()
    shading_elem = shading.makeelement(qn('w:shd'), {
        qn('w:val'): 'clear',
        qn('w:color'): 'auto',
        qn('w:fill'): color,
    })
    shading.append(shading_elem)


def add_table(doc, headers, rows, col_widths=None):
    """添加带样式的表格"""
    table = doc.add_table(rows=1 + len(rows), cols=len(headers))
    table.style = 'Table Grid'
    table.alignment = WD_TABLE_ALIGNMENT.CENTER

    # 表头
    for i, header in enumerate(headers):
        cell = table.rows[0].cells[i]
        cell.text = header
        for p in cell.paragraphs:
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            for run in p.runs:
                run.bold = True
                run.font.size = Pt(10)
                run.font.color.rgb = RGBColor(0xFF, 0xFF, 0xFF)
        set_cell_shading(cell, '2F5496')

    # 数据行
    for r, row in enumerate(rows):
        for c, val in enumerate(row):
            cell = table.rows[r + 1].cells[c]
            cell.text = str(val)
            for p in cell.paragraphs:
                for run in p.runs:
                    run.font.size = Pt(10)
            if r % 2 == 1:
                set_cell_shading(cell, 'D6E4F0')

    if col_widths:
        for i, w in enumerate(col_widths):
            for row in table.rows:
                row.cells[i].width = Cm(w)

    doc.add_paragraph()
    return table


def generate():
    doc = Document()

    # ===== 全局样式 =====
    style = doc.styles['Normal']
    style.font.name = '宋体'
    style.font.size = Pt(12)
    style.paragraph_format.line_spacing = 1.5
    style.element.rPr.rFonts.set(qn('w:eastAsia'), '宋体')

    for level in range(1, 4):
        hs = doc.styles[f'Heading {level}']
        hs.font.color.rgb = RGBColor(0x2F, 0x54, 0x96)
        hs.font.bold = True
        hs.element.rPr.rFonts.set(qn('w:eastAsia'), '黑体')
        if level == 1:
            hs.font.size = Pt(22)
        elif level == 2:
            hs.font.size = Pt(16)
        else:
            hs.font.size = Pt(14)

    # ===== 封面 =====
    for _ in range(6):
        doc.add_paragraph()

    title = doc.add_paragraph()
    title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = title.add_run('ImageStyleTransfer')
    run.font.size = Pt(36)
    run.font.bold = True
    run.font.color.rgb = RGBColor(0x2F, 0x54, 0x96)

    subtitle = doc.add_paragraph()
    subtitle.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = subtitle.add_run('基于风格迁移的视觉隐私保护系统')
    run.font.size = Pt(22)
    run.font.color.rgb = RGBColor(0x2F, 0x54, 0x96)

    doc.add_paragraph()

    desc = doc.add_paragraph()
    desc.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = desc.add_run('中国计算机设计大赛 参赛作品')
    run.font.size = Pt(16)
    run.font.color.rgb = RGBColor(0x59, 0x56, 0x59)

    doc.add_page_break()

    # ===== 目录页 =====
    doc.add_heading('目录', level=1)
    toc_items = [
        '一、项目概述',
        '    1.1 项目背景',
        '    1.2 核心理念',
        '    1.3 系统定位',
        '二、核心功能详解',
        '    2.1 实时摄像头风格迁移',
        '    2.2 多风格自由切换',
        '    2.3 多后端推理引擎',
        '    2.4 一键拍照与实时录像',
        '    2.5 风格强度灵活调节',
        '    2.6 跳帧优化与性能调优',
        '三、技术架构',
        '    3.1 系统架构',
        '    3.2 技术栈',
        '    3.3 模型规格',
        '    3.4 性能实测数据',
        '四、隐私保护机制深度解析',
        '    4.1 "可辨识性"与"可识别性"的分离',
        '    4.2 与传统方案的对比',
        '    4.3 隐私保护强度可调',
        '    4.4 边缘计算的安全优势',
        '五、应用场景',
        '六、目标客户',
        '七、商业价值',
        '八、技术实现亮点',
        '九、部署与运维',
        '十、总结',
    ]
    for item in toc_items:
        p = doc.add_paragraph(item)
        p.paragraph_format.space_before = Pt(2)
        p.paragraph_format.space_after = Pt(2)
        for run in p.runs:
            run.font.size = Pt(12)

    doc.add_page_break()

    # ================================================================
    # 一、项目概述
    # ================================================================
    doc.add_heading('一、项目概述', level=1)

    doc.add_heading('1.1 项目背景', level=2)
    doc.add_paragraph(
        '在当今社会，摄像头无处不在：学校教室、幼儿园活动区、公司办公区、'
        '公共场所的监控录像每天产生海量视频数据。一旦这些数据被不法分子窃取或意外泄露，'
        '将直接暴露个人面部特征、行为轨迹乃至敏感场所的内部布局。'
    )
    doc.add_paragraph(
        '中国《个人信息保护法》（2021）、《数据安全法》（2021）以及欧盟 GDPR '
        '等法规对个人影像数据的保护提出了严格要求，违规处罚力度最高可达年营业额的 4%。'
        '传统的隐私保护手段——如模糊、马赛克——虽然简单直接，但存在明显缺陷：'
        '画面信息被彻底破坏，丧失了可辨识性和实用价值，且马赛克本身容易被识别为'
        '"有敏感内容"，反而引起关注。'
    )

    doc.add_heading('1.2 核心理念', level=2)
    doc.add_paragraph(
        '本项目提出了一种"保留可辨识性、消除可识别性"的隐私保护新范式。'
        '通过将视频/图像进行风格迁移处理，使画面在保持"可辨识性"的同时丧失'
        '"可识别性"——即熟悉画面中人物的人仍能大致认出是谁，但陌生人无法通过画面'
        '识别出具体身份信息，从而在视频泄露场景下最大程度保护个人隐私。'
    )
    doc.add_paragraph(
        '换言之，风格迁移后的视频从"高清监控画面"变成了"艺术风格画面"。'
        '熟悉的人看了大概能认出来是谁、东西大概的样子也能看出来，'
        '但不熟悉的人看了也认不出具体身份。即便视频泄露，也不会造成严重的隐私损失。'
    )

    doc.add_heading('1.3 系统定位', level=2)
    doc.add_paragraph(
        'ImageStyleTransfer 是一套基于深度学习风格迁移技术的视觉隐私保护系统，'
        '运行在 NVIDIA Jetson Orin Nano 边缘计算平台上。系统接入本地摄像头，'
        '对每一帧画面进行实时风格迁移处理，支持多种风格自由切换、一键拍照、实时录像等功能。'
        '整个处理过程在设备端完成，原始视频数据不出设备，从根本上避免了云端传输带来的二次泄露风险。'
    )

    doc.add_page_break()

    # ================================================================
    # 二、核心功能详解
    # ================================================================
    doc.add_heading('二、核心功能详解', level=1)

    doc.add_heading('2.1 实时摄像头风格迁移', level=2)
    doc.add_paragraph(
        '系统接入本地摄像头（CSI 接口 IMX219 或 USB 摄像头），对每一帧画面进行实时风格迁移处理，'
        'OpenCV 窗口左右并排显示原始画面与风格化结果。用户可以直观地看到隐私保护效果。'
    )

    p = doc.add_paragraph()
    p.add_run('技术实现：').bold = True
    bullets = [
        '基于 GStreamer 的硬件加速摄像头采集管道（app/camera/csi_camera.py）',
        'nvarguscamerasrc 驱动直接读取 CSI 传感器原始数据',
        'nvvidconv 硬件色彩空间转换（NVMM 显存直传，零 CPU 开销）',
        '支持硬件缩放：GStreamer 管道中直接将画面缩放到模型输入尺寸',
        '多线程架构：GStreamer 采集线程与主推理线程分离',
        '实测帧率约 13 FPS（Jetson Orin Super Nano，TensorRT FP16，256×256 输入）',
    ]
    for b in bullets:
        doc.add_paragraph(b, style='List Bullet')

    doc.add_heading('2.2 多风格自由切换', level=2)
    doc.add_paragraph(
        '系统预置 3 种风格迁移模型，覆盖不同艺术风格，用户可随时切换：'
    )

    add_table(doc,
        ['风格 ID', '风格名称', '模型文件', '视觉效果'],
        [
            ['1', '🎨 动漫风格', 'animegan_v2.onnx', '清新日系动漫画风，线条清晰'],
            ['2', '🖌️ 人脸动漫', 'animegan_v2_face_paint.onnx', '专注人脸的动漫化处理'],
            ['3', '🎭 帕普莉卡', 'animegan_v2_paprika.onnx', '今敏《帕普莉卡》色彩风格'],
        ]
    )

    doc.add_paragraph(
        '操作方式：按键 1、2、3 即可实时切换风格，无需重启程序。切换瞬间完成，不影响实时预览流畅度。'
    )

    p = doc.add_paragraph()
    p.add_run('技术实现：').bold = True
    bullets = [
        '风格配置文件化管理（config/styles.json），每个风格独立配置模型路径、输入尺寸、归一化方式、锐化强度等参数',
        'StyleManager 负责加载和管理所有风格配置',
        'MultiBackendEngine 支持热切换：相同模型仅更新配置引用（零开销），不同时自动卸载旧模型并加载新模型',
        '用户可自行添加新风格：将 GAN 风格迁移的 ONNX 模型放入 models/ 目录，在 config/styles.json 中添加配置即可',
    ]
    for b in bullets:
        doc.add_paragraph(b, style='List Bullet')

    doc.add_heading('2.3 多后端推理引擎', level=2)
    doc.add_paragraph('系统支持两种推理后端，自动检测并选择最优方案：')

    add_table(doc,
        ['后端', '速度', '依赖', '适用场景'],
        [
            ['TensorRT FP16', '~61.5ms/帧', 'TensorRT（Jetson 预装）', 'GPU 加速，推荐生产环境'],
            ['ONNX Runtime CPU', '~15ms/帧', 'onnxruntime', '无需 GPU，兼容性好'],
        ]
    )

    p = doc.add_paragraph()
    p.add_run('技术实现：').bold = True
    bullets = [
        'MultiBackendEngine 自动检测可用后端，优先级：TensorRT > ONNX Runtime',
        'TensorRT 后端使用原生 TensorRT Python API，PyTorch CUDA 张量管理 GPU 内存',
        'execute_async_v3 异步推理 + torch.cuda.Stream 流同步',
        '支持通过命令行参数 --backend tensorrt/onnx/auto 手动指定或自动选择',
    ]
    for b in bullets:
        doc.add_paragraph(b, style='List Bullet')

    doc.add_heading('2.4 一键拍照与实时录像', level=2)
    bullets = [
        '拍照功能：按空格键一键保存当前画面，同时保存原图、风格化图和左右对比图，文件名自带时间戳',
        '录像功能：按 R 键开始/停止录像，录制左右对比视频（左原图、右风格化），支持 MP4/AVI 格式',
        '鼠标交互：屏幕右上角绘制 REC/STOP 按钮，支持鼠标点击控制录像',
        '录像期间屏幕叠加红色 "REC" 标识和录像时长',
    ]
    for b in bullets:
        doc.add_paragraph(b, style='List Bullet')

    doc.add_heading('2.5 风格强度灵活调节', level=2)
    doc.add_paragraph('系统支持通过 --strength 参数调节风格化程度：')

    add_table(doc,
        ['strength 值', '效果', '推理次数', '隐私保护强度'],
        [
            ['0.1 ~ 0.9', '与原图混合，效果减弱', '1 次', '低（偏真实）'],
            ['1.0', '标准风格化', '1 次', '中'],
            ['1.1 ~ 2.0', '双重推理叠加，效果增强', '2 次', '高（偏抽象）'],
        ]
    )

    doc.add_paragraph(
        '设计意义：不同场景对隐私保护强度的需求不同。幼儿园可能需要较强的风格化以保护儿童身份，'
        '而企业管理场景可能只需轻度风格化以保留更多场景信息。用户可根据实际需求灵活配置。'
    )

    doc.add_heading('2.6 跳帧优化与性能调优', level=2)
    bullets = [
        '跳帧模式（--skip-frames）：每隔 N 帧才执行一次推理，其余帧复用上一次结果，视觉更流畅',
        '锐化补偿（sharpen 参数）：GAN 输出通常偏模糊，系统使用 Unsharp Mask 锐化算法补偿',
        '硬件缩放：GStreamer 管道中直接将摄像头画面缩放到模型输入尺寸，省去 CPU 层面的 resize 开销',
        '性能模式：支持 nvpmodel -m 0 和 jetson_clocks 锁频，最大化 GPU 性能',
    ]
    for b in bullets:
        doc.add_paragraph(b, style='List Bullet')

    doc.add_page_break()

    # ================================================================
    # 三、技术架构
    # ================================================================
    doc.add_heading('三、技术架构', level=1)

    doc.add_heading('3.1 系统架构', level=2)

    # 用文本框展示架构图
    arch_text = """┌─────────────────────────────────────────────────────────┐
│                   OpenCV 显示窗口                        │
│         (原始画面 | 风格化画面 并排对比)                   │
└──────────────┬──────────────────────┬───────────────────┘
               │                      │
       ┌───────▼───────┐      ┌──────▼───────┐
       │   CSICamera    │      │ StorageManager│
       │ (GStreamer采集) │      │ (存储/导出)   │
       └───────┬───────┘      └──────────────┘
               │
       ┌───────▼────────────────────────────┐
       │      StyleTransferApp (主控)        │
       │  帧采集 → 跳帧 → 推理 → 显示/录像   │
       └───────┬────────────────────────────┘
               │
       ┌───────▼────────────────────────────┐
       │      MultiBackendEngine             │
       │  (多后端推理: TensorRT / ONNX RT)   │
       └───────┬────────────────────────────┘
               │
       ┌───────▼────────────────────────────┐
       │         GANEngine                   │
       │  预处理 → 推理 → 后处理 → 强度调节   │
       └───────┬────────────────────────────┘
               │
       ┌───────▼────────────────────────────┐
       │         StyleManager                │
       │  (风格配置: config/styles.json)      │
       └────────────────────────────────────┘"""

    p = doc.add_paragraph()
    run = p.add_run(arch_text)
    run.font.name = 'Consolas'
    run.font.size = Pt(9)

    doc.add_heading('3.2 技术栈', level=2)

    add_table(doc,
        ['层级', '技术选型', '说明'],
        [
            ['硬件平台', 'NVIDIA Jetson Orin Nano 8GB', '边缘 AI 计算平台，功耗 7-15W'],
            ['摄像头驱动', 'GStreamer + nvarguscamerasrc', 'CSI 硬件采集，NVMM 显存直传'],
            ['AI 模型', 'AnimeGANv2（GAN 生成对抗网络）', '轻量级风格迁移，参数量 ~8M'],
            ['推理加速', 'TensorRT FP16（原生 Python API）', 'GPU 加速，PyTorch CUDA 内存管理'],
            ['备选推理', 'ONNX Runtime CPU', '通用兼容方案'],
            ['图像处理', 'OpenCV 4.8+, NumPy', '采集、显示、预处理、后处理'],
            ['模型格式', 'ONNX（开放神经网络交换格式）', '跨框架通用，便于部署'],
            ['数据存储', '文件系统 + 外部存储同步', 'SD 卡 / USB 自动导出'],
            ['系统优化', 'nvpmodel + jetson_clocks', 'GPU 锁频、性能模式、风扇调速'],
        ]
    )

    doc.add_heading('3.3 模型规格', level=2)

    add_table(doc,
        ['指标', '数值'],
        [
            ['模型架构', 'AnimeGANv2 生成器'],
            ['参数量', '~8M'],
            ['模型大小', '~8.5MB（ONNX），~5MB（TensorRT Engine）'],
            ['输入格式', '1×3×H×W float32'],
            ['输出格式', '1×3×H×W float32'],
            ['显存占用', '~200-500MB'],
            ['推理方式', '单次前向传播（非迭代式）'],
        ]
    )

    doc.add_heading('3.4 性能实测数据', level=2)
    doc.add_paragraph(
        '基于 Jetson Orin Super Nano 8GB，TensorRT FP16，256×256 输入，实测数据：'
    )

    p = doc.add_paragraph()
    p.add_run('每帧处理流程耗时：').bold = True

    add_table(doc,
        ['步骤', '平均耗时', '占比', '说明'],
        [
            ['摄像头采集', '~5ms', '7%', 'GStreamer 硬件采集'],
            ['预处理', '~2.2ms', '3.3%', 'BGR→RGB, Resize, 归一化'],
            ['GAN 推理', '~61.5ms', '91.9%', 'TensorRT FP16，主要瓶颈'],
            ['后处理', '~2.5ms', '3.7%', '反归一化, RGB→BGR'],
            ['缩放回原尺寸', '~1.3ms', '1.9%', '模型尺寸 → 显示尺寸'],
            ['显示渲染', '~5ms', '7%', 'OpenCV 窗口渲染'],
            ['整帧合计', '~78ms', '100%', '—'],
            ['实际帧率', '~13 FPS', '—', '满足实时预览需求'],
        ]
    )

    p = doc.add_paragraph()
    p.add_run('资源占用：').bold = True

    add_table(doc,
        ['资源', '用途', '占用量'],
        [
            ['GPU', 'GAN 推理（TensorRT/CUDA）', '推理期间 90%+，显存 200-500MB'],
            ['CPU', '预处理/后处理/显示/录像编码', '~1-2 核'],
            ['内存', '帧缓冲（原始帧 + 风格化帧 + 拼接帧）', '~15-30MB'],
            ['功耗', '整机', '7-15W'],
        ]
    )

    p = doc.add_paragraph()
    p.add_run('瓶颈分析：').bold = True
    doc.add_paragraph(
        'GAN 推理占整帧时间的 79%，是唯一的性能瓶颈。系统通过跳帧机制、'
        '硬件缩放、TensorRT FP16 加速、按需推理等手段优化整体体验。'
    )

    doc.add_page_break()

    # ================================================================
    # 四、隐私保护机制深度解析
    # ================================================================
    doc.add_heading('四、隐私保护机制深度解析', level=1)

    doc.add_heading('4.1 "可辨识性"与"可识别性"的分离', level=2)
    doc.add_paragraph(
        '传统隐私保护（模糊、马赛克）是同时消除"可辨识性"和"可识别性"的——'
        '画面被彻底破坏，谁都看不懂。本项目的核心创新在于只消除"可识别性"，保留"可辨识性"：'
    )

    add_table(doc,
        ['概念', '定义', '风格迁移后的状态'],
        [
            ['可辨识性', '熟悉的人能认出"是谁"', '✅ 保留——家人、同事仍能辨认'],
            ['可识别性', '陌生人能识别具体身份', '❌ 消除——无法提取面部特征、无法 OCR'],
        ]
    )

    doc.add_paragraph('这意味着：')
    bullets = [
        '家人/同事看风格化视频：能认出"这是我们办公室""这是小王"——满足知情需求',
        '不法分子获取风格化视频：无法提取人脸特征、无法进行人脸识别匹配——隐私得到保护',
    ]
    for b in bullets:
        doc.add_paragraph(b, style='List Bullet')

    doc.add_heading('4.2 与传统方案的对比', level=2)

    add_table(doc,
        ['方案', '隐私保护', '画面可用性', '美观度', '是否可逆', '计算成本'],
        [
            ['模糊', '中', '低', '差', '否', '极低'],
            ['马赛克', '中', '低', '差', '否', '低'],
            ['人脸检测+遮挡', '高', '中', '一般', '否', '中'],
            ['风格迁移（本项目）', '高', '高', '好', '否', '中'],
        ]
    )

    doc.add_heading('4.3 隐私保护强度可调', level=2)
    doc.add_paragraph('通过 --strength 参数，用户可以在"保留更多原始信息"和"更强隐私保护"之间灵活调节：')
    bullets = [
        'strength = 0.5：轻度风格化，保留大量原始细节，适合内部管理场景',
        'strength = 1.0：标准风格化，平衡隐私保护与可辨识性',
        'strength = 1.5~2.0：强风格化，面部特征被深度覆盖，适合高安全等级场景',
    ]
    for b in bullets:
        doc.add_paragraph(b, style='List Bullet')

    doc.add_heading('4.4 边缘计算的安全优势', level=2)
    doc.add_paragraph('所有处理均在 Jetson 设备端完成，原始视频数据不出设备：')
    bullets = [
        '无云端传输 → 无法被网络嗅探截获',
        '无云端存储 → 无服务器泄露风险',
        '风格化后才存储/传输 → 即使存储介质被盗，也无法获取原始画面',
        '设备物理隔离 → 满足涉密场所的数据本地化要求',
    ]
    for b in bullets:
        doc.add_paragraph(b, style='List Bullet')

    doc.add_page_break()

    # ================================================================
    # 五、应用场景
    # ================================================================
    doc.add_heading('五、应用场景', level=1)

    scenarios = [
        ('5.1 学校与教育机构', '教室监控录像包含大量学生的面部信息和行为记录。一旦泄露，涉及未成年人隐私保护的法律风险极高。',
         '在监控系统前端部署风格迁移模块，所有存储和传输的视频均为风格化版本。教师和管理人员通过授权查看原始画面，而外部人员即使获取到录像，也无法识别具体学生身份。',
         '满足法律合规要求，降低学校法律风险，同时家长通过风格化视频仍能了解孩子在校活动情况。'),

        ('5.2 幼儿园', '幼儿园监控涉及大量低龄儿童，家长对隐私保护高度敏感。幼儿园需要向家长展示日常活动，但又不希望视频被二次传播后暴露孩子身份。',
         '对面向家长公开的视频进行风格化处理。家长能认出自家孩子，但无法从视频中提取其他孩子的面部信息。',
         '平衡家长知情权与儿童隐私保护，提升幼儿园品牌信任度。'),

        ('5.3 企业与公司', '办公场所监控涉及员工行为轨迹、屏幕内容、会议画面等敏感信息。',
         '办公监控轻度风格化处理；会议录像风格化后存档；访客登记照片风格化存储。',
         '降低企业数据泄露的法律和商业风险，符合《数据安全法》《个人信息保护法》合规要求。'),

        ('5.4 机密场所与政府机构', '军事设施、科研实验室、政府机关等场所的监控视频一旦泄露，可能涉及国家安全。',
         '使用风格迁移替代传统马赛克。风格化后的视频看起来像"艺术作品"，既保护敏感信息，又不会引起额外关注。',
         '提供更隐蔽、更优雅的视觉信息保护手段。边缘部署确保数据不出设备。'),

        ('5.5 公共场所与智慧城市', '城市公共摄像头覆盖广泛，录像数据量巨大。数据集中存储面临泄露风险。',
         '在摄像头端进行实时风格迁移处理。原始视频仅在授权终端实时查看、不落盘；落盘存储的均为风格化版本。',
         '在保障公共安全的同时保护公民隐私，推动智慧城市建设中的隐私合规。'),

        ('5.6 医疗与健康机构', '医疗影像、患者监控视频涉及高度敏感的个人健康信息。',
         '对医疗监控视频进行风格化处理，保留医护人员的操作动作和流程信息，同时隐去患者面部特征。',
         '满足医疗行业严格的隐私保护法规要求。'),

        ('5.7 内容创作与媒体', '新闻采访、纪录片拍摄中，可能涉及不愿露面的受访者或需要匿名处理的人物。',
         '对视频素材进行实时或后期风格化处理，在保留画面叙事性的同时保护人物身份。',
         '为内容创作者提供美观、自然的匿名处理方式。'),
    ]

    for title, pain, solution, value in scenarios:
        doc.add_heading(title, level=2)
        p = doc.add_paragraph()
        p.add_run('痛点：').bold = True
        p.add_run(pain)

        p = doc.add_paragraph()
        p.add_run('方案：').bold = True
        p.add_run(solution)

        p = doc.add_paragraph()
        p.add_run('价值：').bold = True
        p.add_run(value)

    doc.add_page_break()

    # ================================================================
    # 六、目标客户
    # ================================================================
    doc.add_heading('六、目标客户', level=1)

    add_table(doc,
        ['客户群体', '典型客户', '核心需求', '付费意愿'],
        [
            ['教育行业', '中小学、幼儿园、培训机构', '未成年人隐私保护、家长沟通', '中等（合规驱动）'],
            ['企业客户', '科技公司、金融机构、制造企业', '员工隐私、商业机密保护', '高（风险驱动）'],
            ['政府机关', '公安、国安、军事单位', '涉密场所信息保护', '高（安全驱动）'],
            ['物业管理', '写字楼、商场、住宅小区', '公共区域监控隐私合规', '中等（合规驱动）'],
            ['医疗健康', '医院、养老院、康复中心', '患者隐私保护', '高（法规驱动）'],
            ['安防行业', '监控设备厂商、安防集成商', '为产品增加隐私保护能力', '高（差异化驱动）'],
            ['内容创作者', '视频博主、纪录片团队', '素材中人脸的隐私处理', '低（个人用户）'],
        ]
    )

    doc.add_page_break()

    # ================================================================
    # 七、商业价值
    # ================================================================
    doc.add_heading('七、商业价值', level=1)

    doc.add_heading('7.1 市场背景与机遇', level=2)

    p = doc.add_paragraph()
    p.add_run('法规驱动：').bold = True
    bullets = [
        '中国《个人信息保护法》（2021）明确将"面部识别信息"列为敏感个人信息',
        '《数据安全法》（2021）建立数据分类分级保护制度',
        '欧盟 GDPR 对个人影像数据保护要求严格，违规处罚最高可达年营业额 4%',
    ]
    for b in bullets:
        doc.add_paragraph(b, style='List Bullet')

    p = doc.add_paragraph()
    p.add_run('市场规模：').bold = True
    bullets = [
        '全球视频监控市场规模超过 500 亿美元，年增长率约 10%',
        '隐私保护作为合规刚需，是其中增长最快的细分领域',
        '边缘 AI 市场预计 2025 年达到 150 亿美元',
    ]
    for b in bullets:
        doc.add_paragraph(b, style='List Bullet')

    doc.add_heading('7.2 核心商业价值', level=2)

    values = [
        ('合规价值 — 降低法律风险',
         '帮助企业和机构满足《个人信息保护法》《数据安全法》等法规对视频数据的保护要求。'
         '避免因数据泄露导致的巨额罚款、诉讼赔偿和声誉损失。'),
        ('差异化竞争力 — 产品功能增强',
         '对于安防厂商，集成隐私保护能力可作为产品卖点。对于 SaaS 平台，隐私保护功能可提升客户信任度和付费意愿。'),
        ('边缘部署能力 — 降低成本',
         '基于 Jetson 边缘设备，单设备功耗仅 7-15W，运行成本极低。无需云端算力，无网络带宽需求。'),
        ('灵活可配置 — 适配多场景',
         '3 种预设风格覆盖不同视觉偏好。风格强度可调（0.1~2.0），适配不同隐私保护等级。'),
    ]
    for title, desc in values:
        p = doc.add_paragraph()
        p.add_run(f'（{values.index((title, desc)) + 1}）{title}').bold = True
        doc.add_paragraph(desc)

    doc.add_heading('7.3 商业模式', level=2)

    add_table(doc,
        ['模式', '说明', '目标客户'],
        [
            ['硬件整机销售', 'Jetson 设备 + 摄像头 + 外壳 + 预装软件，开箱即用', '学校、幼儿园、中小企业'],
            ['软件授权', '按摄像头数量/并发路数收取年度授权费', '大型企业、安防集成商'],
            ['SDK/OEM 集成', '向安防厂商提供 SDK，按出货量收取授权费', '监控设备厂商'],
            ['定制开发', '为大型客户提供定制化风格模型和功能开发', '政府机关、军工单位'],
            ['SaaS 服务', '云端 API 按调用次数/时长计费', '内容创作者、中小企业'],
        ]
    )

    doc.add_heading('7.4 竞争优势', level=2)

    add_table(doc,
        ['优势', '说明'],
        [
            ['非破坏性保护', '风格迁移不同于马赛克/模糊，画面仍有观赏性和辨识度'],
            ['边缘实时处理', '13+ FPS 实时处理，满足监控场景的实时性要求'],
            ['低功耗低成本', '7-15W 功耗，千元级设备成本，适合大规模部署'],
            ['数据不出设备', '边缘计算天然满足数据本地化合规要求'],
            ['灵活可配置', '多风格、多强度、多模型，适配不同场景和安全等级'],
            ['开源可控', '核心代码自主可控，不依赖第三方隐私服务'],
            ['模型轻量', '单模型仅 ~8.5MB，3 个模型总计不到 30MB'],
        ]
    )

    doc.add_heading('7.5 市场拓展路径', level=2)

    phases = [
        ('第一阶段（0-6个月）：教育行业切入', '幼儿园/中小学试点部署，打磨产品稳定性，建立教育行业标杆案例'),
        ('第二阶段（6-12个月）：企业市场拓展', '企业办公监控场景，安防厂商 SDK 合作，物业管理场景覆盖'),
        ('第三阶段（12-24个月）：政府与行业深耕', '政府机关/涉密场所，医疗行业合规方案，智慧城市整体方案'),
    ]
    for title, desc in phases:
        p = doc.add_paragraph()
        p.add_run(title).bold = True
        doc.add_paragraph(desc, style='List Bullet')

    doc.add_page_break()

    # ================================================================
    # 八、技术实现亮点
    # ================================================================
    doc.add_heading('八、技术实现亮点', level=1)

    highlights = [
        ('8.1 原生 TensorRT 推理',
         '使用原生 TensorRT Python API 进行推理，充分发挥 Jetson GPU 性能。'
         '包括 trt.Runtime 反序列化引擎、execute_async_v3 异步推理、PyTorch CUDA 张量管理 GPU 内存等。'),
        ('8.2 GStreamer 硬件加速采集',
         '摄像头采集完全绕过 OpenCV 的 GStreamer 后端，使用原生 GStreamer Python 绑定。'
         'nvarguscamerasrc 直接读取 CSI 传感器，nvvidconv 硬件色彩空间转换，零 CPU 开销。'),
        ('8.3 智能模型切换',
         'MultiBackendEngine 实现智能模型切换：相同模型的不同风格切换零开销，'
         '不同模型自动卸载/加载，后端自动检测优先级 TensorRT > ONNX Runtime。'),
        ('8.4 双重推理强度调节',
         'GANEngine.transfer() 实现灵活的强度调节：strength < 1.0 与原图混合，'
         'strength = 1.0 标准推理，strength > 1.0 双重推理叠加。'),
        ('8.5 Unsharp Mask 锐化补偿',
         'GAN 模型输出通常偏模糊，系统使用 Unsharp Mask 算法进行锐化补偿，'
         '强度可通过 config/styles.json 中的 sharpen 参数配置。'),
    ]
    for title, desc in highlights:
        doc.add_heading(title, level=2)
        doc.add_paragraph(desc)

    doc.add_page_break()

    # ================================================================
    # 九、部署与运维
    # ================================================================
    doc.add_heading('九、部署与运维', level=1)

    doc.add_heading('9.1 一键部署流程', level=2)

    code_text = """# 1. 准备模型（Windows 上操作）
python scripts/prepare_models.py

# 2. 传代码到 Jetson
scp -r * user@<IP>:~/jetson-style-transfer/

# 3. 转换 TensorRT 引擎
python3 scripts/convert_tensorrt.py --all

# 4. 启动应用
python3 -m app.main --backend tensorrt"""

    p = doc.add_paragraph()
    run = p.add_run(code_text)
    run.font.name = 'Consolas'
    run.font.size = Pt(10)

    doc.add_heading('9.2 开机自启动', level=2)
    doc.add_paragraph('系统支持配置开机自启动，适合无人值守场景。通过创建 ~/.config/autostart/style-transfer.desktop 文件实现。')

    doc.add_heading('9.3 远程运维', level=2)
    bullets = [
        'SSH 远程连接管理',
        'tegrastats 实时监控 GPU/CPU 使用率和温度',
        '日志系统记录推理性能、错误信息',
        '支持远程更新模型和配置',
    ]
    for b in bullets:
        doc.add_paragraph(b, style='List Bullet')

    doc.add_page_break()

    # ================================================================
    # 十、总结
    # ================================================================
    doc.add_heading('十、总结', level=1)

    doc.add_paragraph(
        'ImageStyleTransfer 通过将深度学习风格迁移技术应用于视觉隐私保护，'
        '提出了一种"保留可辨识性、消除可识别性"的隐私保护新范式。'
        '系统基于 NVIDIA Jetson Orin Nano 边缘计算平台，集成了 AnimeGANv2 风格迁移模型，'
        '支持 TensorRT GPU 加速实时推理，提供多风格切换、强度调节、拍照录像等丰富功能。'
    )

    doc.add_paragraph('与传统隐私保护方案相比，本系统具有以下独特优势：')
    bullets = [
        '隐私保护与可用性的平衡：风格化画面既有艺术美感，又保留了足够的场景信息',
        '边缘计算安全保障：数据不出设备，从根本上避免云端泄露风险',
        '低功耗低成本：7-15W 功耗、千元级设备成本，适合大规模部署',
        '灵活可配置：多风格、多强度、多后端，适配不同场景需求',
    ]
    for b in bullets:
        doc.add_paragraph(b, style='List Bullet')

    doc.add_paragraph(
        '在隐私法规日趋严格、公众隐私意识不断提升的大背景下，本系统具有明确的市场需求和商业价值，'
        '尤其在教育、安防、政府等强监管行业具有广阔的应用前景。'
    )

    # ===== 保存 =====
    output_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                               'docs', 'ImageStyleTransfer_项目说明文档.docx')
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    doc.save(output_path)
    print(f'文档已生成: {output_path}')


if __name__ == '__main__':
    generate()

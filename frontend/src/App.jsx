import React, { useState, useRef, useCallback, useEffect } from 'react'

const API = 'http://localhost:8000'  // 直连后端

export default function App() {
  const [image, setImage] = useState(null)         // {id, width, height, url}
  const [tool, setTool] = useState('point')         // point | box | brush
  const [points, setPoints] = useState([])          // [{x, y, label}]
  const [box, setBox] = useState(null)              // {x1, y1, x2, y2} | null
  const [isDrawing, setIsDrawing] = useState(false)
  const [result, setResult] = useState(null)
  const [loading, setLoading] = useState(false)
  const [dragOver, setDragOver] = useState(false)
  const [health, setHealth] = useState(null)
  const [detectResult, setDetectResult] = useState(null)
  const [recognizeResult, setRecognizeResult] = useState(null)
  const [autoSegResult, setAutoSegResult] = useState(null)
  const [colorObjects, setColorObjects] = useState(null)
  const [selectedColorObj, setSelectedColorObj] = useState(null)
  const [theme, setTheme] = useState('dark')        // dark | light
  
  // 模型切换状态
  const [currentModel, setCurrentModel] = useState('vit_b')
  const [availableModels, setAvailableModels] = useState([])
  const [switchingModel, setSwitchingModel] = useState(false)
  
  // 掩码编辑状态
  const [brushMode, setBrushMode] = useState('add')  // add | erase
  const [brushSize, setBrushSize] = useState(15)
  const [currentMask, setCurrentMask] = useState(null)  // base64
  const [brushStrokes, setBrushStrokes] = useState([])
  
  // 分割历史记录状态
  const [history, setHistory] = useState([])
  const [selectedHistoryId, setSelectedHistoryId] = useState(null)
  const [toolbarExpanded, setToolbarExpanded] = useState(true)
  const [isMobile, setIsMobile] = useState(window.innerWidth <= 768)
  
  // 视频/摄像头状态
  const [videoMode, setVideoMode] = useState('image')  // image | video | camera
  const [videoData, setVideoData] = useState(null)  // {video_id, filename, info, frames}
  const [currentFrame, setCurrentFrame] = useState(null)  // 当前选中的帧
  const [frameIndex, setFrameIndex] = useState(0)  // 当前帧索引
  const [cameras, setCameras] = useState([])  // 可用摄像头列表
  const [selectedCamera, setSelectedCamera] = useState(0)  // 选中的摄像头
  const [cameraStreamUrl, setCameraStreamUrl] = useState(null)  // 摄像头流 URL
  
  // 批量图片状态
  const [batchImages, setBatchImages] = useState([])  // 批量上传的图片列表
  const [currentBatchIndex, setCurrentBatchIndex] = useState(0)  // 当前显示的图片索引
  
  // 原图备份（用于恢复）
  const [originalImage, setOriginalImage] = useState(null)  // 原始图片数据

  const canvasRef = useRef(null)
  const overlayRef = useRef(null)
  const fileInputRef = useRef(null)
  const imgRef = useRef(null)

  // ── Health check ──
  useEffect(() => {
    fetch(`${API}/api/health`).then(r => r.json()).then(setHealth).catch(() => {})
    // 获取可用模型
    fetch(`${API}/api/models`).then(r => r.json()).then(data => {
      setAvailableModels(data.models || [])
      setCurrentModel(data.current || 'vit_b')
    }).catch(() => {})
  }, [])

  // ── Mobile detection ──
  useEffect(() => {
    const handleResize = () => {
      setIsMobile(window.innerWidth <= 768)
    }
    window.addEventListener('resize', handleResize)
    return () => window.removeEventListener('resize', handleResize)
  }, [])

  // ── 切换模型 ──
  const switchSAMModel = useCallback(async (modelType) => {
    if (switchingModel || modelType === currentModel) return
    setSwitchingModel(true)
    try {
      const res = await fetch(`${API}/api/models/switch`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ model_type: modelType })
      })
      const data = await res.json()
      if (data.success) {
        setCurrentModel(modelType)
        // 重新获取模型列表
        const modelsRes = await fetch(`${API}/api/models`)
        const modelsData = await modelsRes.json()
        setAvailableModels(modelsData.models || [])
        alert(`✓ 已切换到 ${modelType.toUpperCase()}`)
      }
    } catch (e) {
      alert('切换模型失败: ' + e.message)
    }
    setSwitchingModel(false)
  }, [currentModel, switchingModel])

  // ── Image upload ──
  const handleUpload = useCallback(async (file) => {
    if (!file) return
    const fd = new FormData()
    fd.append('file', file)
    setLoading(true)
    try {
      const res = await fetch(`${API}/api/upload`, { method: 'POST', body: fd })
      const data = await res.json()
      if (data.image_id) {
        const imgData = {
          id: data.image_id,
          width: data.width,
          height: data.height,
          url: `${API}/api/image/${data.image_id}`
        }
        setImage(imgData)
        setOriginalImage(imgData)  // 保存原图
        setBatchImages([])  // 清空批量图片
        setCurrentBatchIndex(0)
        setPoints([])
        setBox(null)
        setResult(null)
        setColorObjects(null)
      }
    } catch (e) {
      alert('上传失败: ' + e.message)
    }
    setLoading(false)
  }, [])

  const handleDrop = useCallback((e) => {
    e.preventDefault()
    setDragOver(false)
    const file = e.dataTransfer.files[0]
    if (file?.type.startsWith('image/')) handleUpload(file)
  }, [handleUpload])

  // ── Batch upload ──
  const handleBatchUpload = useCallback(async (files) => {
    if (!files || files.length === 0) return
    
    const fd = new FormData()
    for (const file of files) {
      if (file.type.startsWith('image/')) {
        fd.append('files', file)
      }
    }
    
    setLoading(true)
    try {
      const res = await fetch(`${API}/api/upload/batch`, { method: 'POST', body: fd })
      const data = await res.json()
      
      if (data.success && data.images.length > 0) {
        // 保存所有图片
        setBatchImages(data.images)
        setCurrentBatchIndex(0)
        
        // 加载第一张图片
        const first = data.images[0]
        const imgData = {
          id: first.image_id,
          width: first.width,
          height: first.height,
          url: `${API}/api/image/${first.image_id}`
        }
        setImage(imgData)
        setOriginalImage(imgData)  // 保存原图
        setPoints([])
        setBox(null)
        setResult(null)
        setColorObjects(null)
        
        alert(`✓ 成功上传 ${data.count} 张图片！\n当前显示第 1 张，可使用左侧工具栏切换图片。`)
      }
    } catch (e) {
      alert('批量上传失败: ' + e.message)
    }
    setLoading(false)
  }, [])

  // ── 切换批量图片 ──
  const switchBatchImage = useCallback((index) => {
    if (index < 0 || index >= batchImages.length) return
    
    const img = batchImages[index]
    setCurrentBatchIndex(index)
    
    const imgData = {
      id: img.image_id,
      width: img.width,
      height: img.height,
      url: `${API}/api/image/${img.image_id}`
    }
    setImage(imgData)
    setOriginalImage(imgData)
    setPoints([])
    setBox(null)
    setResult(null)
    setCurrentMask(null)
    setColorObjects(null)
  }, [batchImages])

  // ── Fetch history ──
  const fetchHistory = useCallback(async () => {
    if (!image) return
    try {
      const res = await fetch(`${API}/api/history/${image.id}`)
      const data = await res.json()
      if (data.success) {
        setHistory(data.history)
      }
    } catch (e) {
      console.error('获取历史记录失败:', e)
    }
  }, [image])

  // ── Load history entry ──
  const loadHistoryEntry = useCallback(async (entryId) => {
    if (!image) return
    try {
      const res = await fetch(`${API}/api/history/get`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ image_id: image.id, entry_id: entryId })
      })
      const data = await res.json()
      if (data.success && data.entry) {
        const entry = data.entry
        setCurrentMask(entry.mask)
        setResult({
          success: true,
          mask: entry.mask,
          overlay: entry.overlay,
          score: entry.score,
          area: entry.area,
          bbox: entry.bbox
        })
        setSelectedHistoryId(entryId)
      }
    } catch (e) {
      console.error('加载历史记录失败:', e)
    }
  }, [image])

  // ── Delete history entry ──
  const deleteHistoryEntry = useCallback(async (entryId) => {
    if (!image) return
    try {
      const res = await fetch(`${API}/api/history/${image.id}/${entryId}`, {
        method: 'DELETE'
      })
      const data = await res.json()
      if (data.success) {
        fetchHistory()
        if (selectedHistoryId === entryId) {
          setSelectedHistoryId(null)
        }
      }
    } catch (e) {
      console.error('删除历史记录失败:', e)
    }
  }, [image, selectedHistoryId, fetchHistory])

  // ── Clear all history ──
  const clearHistory = useCallback(async () => {
    if (!image) return
    try {
      await fetch(`${API}/api/history/${image.id}`, { method: 'DELETE' })
      setHistory([])
      setSelectedHistoryId(null)
    } catch (e) {
      console.error('清空历史记录失败:', e)
    }
  }, [image])

  // ── 导出历史记录 ──
  const exportHistory = useCallback(async (format) => {
    if (!image) return
    try {
      const url = `${API}/api/export/${format}/${image.id}`
      const response = await fetch(url)
      
      if (!response.ok) {
        throw new Error('导出失败')
      }
      
      const blob = await response.blob()
      const downloadUrl = window.URL.createObjectURL(blob)
      const link = document.createElement('a')
      link.href = downloadUrl
      link.download = `sam_export_${image.id}.${format}`
      document.body.appendChild(link)
      link.click()
      document.body.removeChild(link)
      window.URL.revokeObjectURL(downloadUrl)
    } catch (e) {
      alert('导出失败: ' + e.message)
    }
  }, [image])

  // ── Fetch history when image changes ──
  useEffect(() => {
    if (image) {
      fetchHistory()
    } else {
      setHistory([])
      setSelectedHistoryId(null)
    }
  }, [image, fetchHistory])

  // ── Brush editing ──
  const handleBrushEdit = useCallback(async (x, y) => {
    if (!currentMask || !image) return
    
    // 添加笔触到列表
    const newStrokes = [...brushStrokes, { x, y, radius: brushSize, mode: brushMode }]
    setBrushStrokes(newStrokes)
    
    try {
      const res = await fetch(`${API}/api/mask/edit`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          image_id: image.id,
          mask_base64: currentMask,
          strokes: [{ x, y, radius: brushSize, mode: brushMode }]
        })
      })
      const data = await res.json()
      if (data.success) {
        setCurrentMask(data.mask)
      }
    } catch (e) {
      console.error('画笔编辑失败:', e)
    }
  }, [currentMask, image, brushStrokes, brushSize, brushMode])

  // ── Undo mask edit ──
  const undoMaskEdit = useCallback(async () => {
    if (!image) return
    try {
      const res = await fetch(`${API}/api/mask/undo`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ image_id: image.id })
      })
      const data = await res.json()
      if (data.success) {
        setCurrentMask(data.mask)
      }
    } catch (e) {
      console.error('撤销失败:', e)
    }
  }, [image])

  // ── Redo mask edit ──
  const redoMaskEdit = useCallback(async () => {
    if (!image) return
    try {
      const res = await fetch(`${API}/api/mask/redo`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ image_id: image.id })
      })
      const data = await res.json()
      if (data.success) {
        setCurrentMask(data.mask)
      }
    } catch (e) {
      console.error('重做失败:', e)
    }
  }, [image])

  // ── 视频上传 ──
  const handleVideoUpload = useCallback(async (file) => {
    if (!file) return
    const fd = new FormData()
    fd.append('file', file)
    setLoading(true)
    try {
      const res = await fetch(`${API}/api/video/upload?max_frames=50`, { method: 'POST', body: fd })
      const data = await res.json()
      if (data.video_id) {
        setVideoData(data)
        setVideoMode('video')
        setFrameIndex(0)
        
        // 加载第一帧
        if (data.frames && data.frames.length > 0) {
          const firstFrame = data.frames[0]
          setImage({
            id: firstFrame.image_id,
            width: firstFrame.width,
            height: firstFrame.height,
            url: `${API}/api/image/${firstFrame.image_id}`
          })
          setCurrentFrame(firstFrame)
        }
        
        alert(`✓ 视频上传成功！\n提取了 ${data.frame_count} 帧\n时长: ${data.info.duration?.toFixed(1)}s`)
      }
    } catch (e) {
      alert('视频上传失败: ' + e.message)
    }
    setLoading(false)
  }, [])

  // ── 切换视频帧 ──
  const switchFrame = useCallback(async (index) => {
    if (!videoData || !videoData.frames || index < 0 || index >= videoData.frames.length) return
    
    const frame = videoData.frames[index]
    setFrameIndex(index)
    setCurrentFrame(frame)
    
    // 加载帧图像
    setImage({
      id: frame.image_id,
      width: frame.width,
      height: frame.height,
      url: `${API}/api/image/${frame.image_id}`
    })
    
    // 清除当前分割结果
    setPoints([])
    setBox(null)
    setResult(null)
    setCurrentMask(null)
  }, [videoData])

  // ── 获取可用摄像头 ──
  const fetchCameras = useCallback(async () => {
    try {
      const res = await fetch(`${API}/api/camera/list`)
      const data = await res.json()
      if (data.cameras) {
        setCameras(data.cameras)
      }
    } catch (e) {
      console.error('获取摄像头列表失败:', e)
    }
  }, [])

  // ── 启动摄像头流 ──
  const startCameraStream = useCallback((cameraId) => {
    setSelectedCamera(cameraId)
    setVideoMode('camera')
    setCameraStreamUrl(`${API}/api/camera/${cameraId}/stream`)
  }, [])

  // ── 停止摄像头流 ──
  const stopCameraStream = useCallback(() => {
    setVideoMode('image')
    setCameraStreamUrl(null)
  }, [])

  // ── 获取摄像头列表 ──
  useEffect(() => {
    fetchCameras()
  }, [fetchCameras])

  // ── Draw image on canvas ──
  useEffect(() => {
    if (!image || !canvasRef.current) return
    const canvas = canvasRef.current
    const ctx = canvas.getContext('2d')
    const img = new Image()
    img.crossOrigin = 'anonymous'
    img.onload = () => {
      // Fit to viewport - 考虑工具栏和结果面板的宽度
      const toolbarWidth = 260
      const resultsWidth = 280
      const headerHeight = 56
      const footerHeight = 32
      const padding = 48 // 增加内边距
      
      const maxW = Math.min(window.innerWidth - toolbarWidth - resultsWidth - padding, 700)
      const maxH = Math.min(window.innerHeight - headerHeight - footerHeight - padding, 550)
      
      let w = img.width, h = img.height
      if (w > maxW) { h *= maxW / w; w = maxW }
      if (h > maxH) { w *= maxH / h; h = maxH }
      w = Math.round(w); h = Math.round(h)
      canvas.width = w
      canvas.height = h
      ctx.drawImage(img, 0, 0, w, h)
      imgRef.current = { el: img, dw: w, dh: h, sw: img.width, sh: img.height }
      // Also size overlay
      if (overlayRef.current) {
        overlayRef.current.width = w
        overlayRef.current.height = h
      }
    }
    img.src = image.url
  }, [image])

  // ── Canvas click → point or brush ──
  const handleCanvasClick = useCallback((e) => {
    if (!imgRef.current) return
    const rect = canvasRef.current.getBoundingClientRect()
    const dx = e.clientX - rect.left
    const dy = e.clientY - rect.top
    // Scale to original image coords
    const { dw, dh, sw, sh } = imgRef.current
    const ox = dx * sw / dw
    const oy = dy * sh / dh
    
    if (tool === 'point') {
      const label = e.shiftKey ? 0 : 1  // shift = background
      setPoints(prev => [...prev, { x: ox, y: oy, label, dx, dy }])
    } else if (tool === 'brush' && currentMask) {
      handleBrushEdit(Math.round(ox), Math.round(oy))
    }
  }, [tool, currentMask, handleBrushEdit])

  // ── Touch support for canvas ──
  const handleTouchStart = useCallback((e) => {
    if (!imgRef.current) return
    e.preventDefault() // Prevent scrolling while touching canvas
    const touch = e.touches[0]
    const rect = canvasRef.current.getBoundingClientRect()
    const dx = touch.clientX - rect.left
    const dy = touch.clientY - rect.top
    const { dw, dh, sw, sh } = imgRef.current
    const ox = dx * sw / dw
    const oy = dy * sh / dh
    
    if (tool === 'point') {
      // On touch, add point (no shift key equivalent, use long press in future)
      setPoints(prev => [...prev, { x: ox, y: oy, label: 1, dx, dy }])
    } else if (tool === 'brush' && currentMask) {
      handleBrushEdit(Math.round(ox), Math.round(oy))
    } else if (tool === 'box') {
      setIsDrawing(true)
      setBox({ x1: dx, y1: dy, x2: dx, y2: dy, ox1: ox, oy1: oy })
    }
  }, [tool, currentMask, handleBrushEdit])

  const handleTouchMove = useCallback((e) => {
    if (!imgRef.current) return
    e.preventDefault()
    const touch = e.touches[0]
    const rect = canvasRef.current.getBoundingClientRect()
    const { dw, dh, sw, sh } = imgRef.current
    const dx = Math.max(0, Math.min(touch.clientX - rect.left, dw))
    const dy = Math.max(0, Math.min(touch.clientY - rect.top, dh))
    
    if (tool === 'brush' && currentMask) {
      const ox = dx * sw / dw
      const oy = dy * sh / dh
      handleBrushEdit(Math.round(ox), Math.round(oy))
    } else if (isDrawing && tool === 'box') {
      setBox(prev => prev ? { ...prev, x2: dx, y2: dy, ox2: dx*sw/dw, oy2: dy*sh/dh } : null)
    }
  }, [tool, isDrawing, currentMask, handleBrushEdit])

  const handleTouchEnd = useCallback(() => {
    setIsDrawing(false)
  }, [])

  // ── Box drawing ──
  const handleMouseDown = useCallback((e) => {
    if (tool !== 'box' || !imgRef.current) return
    const rect = canvasRef.current.getBoundingClientRect()
    const { dw, dh, sw, sh } = imgRef.current
    const dx = e.clientX - rect.left
    const dy = e.clientY - rect.top
    setIsDrawing(true)
    setBox({ x1: dx, y1: dy, x2: dx, y2: dy, ox1: dx*sw/dw, oy1: dy*sh/dh })
  }, [tool])

  const handleMouseMove = useCallback((e) => {
    if (!isDrawing || !imgRef.current) return
    const rect = canvasRef.current.getBoundingClientRect()
    const { dw, dh, sw, sh } = imgRef.current
    const dx = Math.max(0, Math.min(e.clientX - rect.left, dw))
    const dy = Math.max(0, Math.min(e.clientY - rect.top, dh))
    setBox(prev => prev ? { ...prev, x2: dx, y2: dy, ox2: dx*sw/dw, oy2: dy*sh/dh } : null)
  }, [isDrawing])

  const handleMouseUp = useCallback(() => {
    setIsDrawing(false)
  }, [])

  // ── Run segmentation ──
  const runSegmentation = useCallback(async () => {
    if (!image) return
    setLoading(true)
    setResult(null)

    try {
      let res
      if (tool === 'point' && points.length > 0) {
        res = await fetch(`${API}/api/segment/point`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            image_id: image.id,
            points: points.map(p => [p.x, p.y]),
            labels: points.map(p => p.label)
          })
        })
      } else if (tool === 'box' && box) {
        res = await fetch(`${API}/api/segment/box`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            image_id: image.id,
            box: [box.ox1, box.oy1, box.ox2, box.oy2]
          })
        })
      } else {
        setLoading(false)
        return
      }

      const data = await res.json()
      setResult(data)

      // Draw overlay (原图+彩色掩码叠加) 直接在主画布上
      if (data.overlay && canvasRef.current && imgRef.current) {
        const ctx = canvasRef.current.getContext('2d')
        const img = new Image()
        img.onload = () => {
          ctx.clearRect(0, 0, canvasRef.current.width, canvasRef.current.height)
          ctx.drawImage(img, 0, 0, canvasRef.current.width, canvasRef.current.height)
        }
        img.src = `data:image/png;base64,${data.overlay}`
        // 清空 overlay canvas
        if (overlayRef.current) {
          overlayRef.current.getContext('2d').clearRect(0, 0, overlayRef.current.width, overlayRef.current.height)
        }
      }
      
      // 刷新历史记录
      if (data.success) {
        fetchHistory()
      }
    } catch (e) {
      setResult({ success: false, message: e.message })
    }
    setLoading(false)
  }, [image, tool, points, box, fetchHistory])

  // ── Clear ──
  const clearAll = useCallback(() => {
    setPoints([])
    setBox(null)
    setResult(null)
    setDetectResult(null)
    setRecognizeResult(null)
    setAutoSegResult(null)
    setColorObjects(null)
    setSelectedColorObj(null)
    setCurrentMask(null)
    setBrushStrokes([])
    // 清空 overlay canvas
    if (overlayRef.current) {
      overlayRef.current.getContext('2d').clearRect(0, 0, overlayRef.current.width, overlayRef.current.height)
    }
    // 重新绘制原图
    if (canvasRef.current && imgRef.current) {
      const ctx = canvasRef.current.getContext('2d')
      const { el, dw, dh } = imgRef.current
      ctx.clearRect(0, 0, dw, dh)
      ctx.drawImage(el, 0, 0, dw, dh)
    }
  }, [])

  // ── Auto Detect (自动检测所有物体) ──
  const runAutoDetect = useCallback(async () => {
    if (!image) return
    setLoading(true)
    setDetectResult(null)

    try {
      const res = await fetch(`${API}/api/detect/auto`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          image_id: image.id,
          points_per_side: 24,
          min_mask_region_area: 200
        })
      })
      const data = await res.json()
      setDetectResult(data)

      // 在画布上显示检测结果
      if (data.overlay && canvasRef.current && imgRef.current) {
        const ctx = canvasRef.current.getContext('2d')
        const img = new Image()
        img.onload = () => {
          ctx.clearRect(0, 0, canvasRef.current.width, canvasRef.current.height)
          ctx.drawImage(img, 0, 0, canvasRef.current.width, canvasRef.current.height)
        }
        img.src = `data:image/png;base64,${data.overlay}`
      }
    } catch (e) {
      setDetectResult({ success: false, message: e.message })
    }
    setLoading(false)
  }, [image])

  // ── Auto Segment (自动分割所有物体，返回每个物体的掩码) ──
  const runAutoSegment = useCallback(async () => {
    if (!image) return
    setLoading(true)
    setAutoSegResult(null)

    try {
      const res = await fetch(`${API}/api/segment/auto`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          image_id: image.id,
          points_per_side: 20,
          min_mask_region_area: 300
        })
      })
      const data = await res.json()
      setAutoSegResult(data)

      // 在画布上显示彩色叠加图
      if (data.overlay && canvasRef.current && imgRef.current) {
        const ctx = canvasRef.current.getContext('2d')
        const img = new Image()
        img.onload = () => {
          ctx.clearRect(0, 0, canvasRef.current.width, canvasRef.current.height)
          ctx.drawImage(img, 0, 0, canvasRef.current.width, canvasRef.current.height)
        }
        img.src = `data:image/png;base64,${data.overlay}`
      }
    } catch (e) {
      setAutoSegResult({ success: false, message: e.message })
    }
    setLoading(false)
  }, [image])

  // ── Recognize (识别图像) ──
  const runRecognize = useCallback(async () => {
    if (!image) return
    setLoading(true)
    setRecognizeResult(null)

    try {
      const res = await fetch(`${API}/api/recognize?image_id=${image.id}`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
      })
      const data = await res.json()
      setRecognizeResult(data)
    } catch (e) {
      setRecognizeResult({ success: false, message: e.message })
    }
    setLoading(false)
  }, [image])

  // ── Extract colored objects (提取彩色物体) ──
  const runExtractColors = useCallback(async () => {
    if (!image) return
    setLoading(true)
    setColorObjects(null)

    try {
      // 使用后端默认置信度阈值 (0.6)
      const res = await fetch(`${API}/api/extract/all?image_id=${image.id}&min_area=500`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
      })
      const data = await res.json()
      
      setColorObjects(data)

      // 在画布上显示彩色叠加图（显示第一个物体或叠加图）
      if (data.objects && data.objects.length > 0 && canvasRef.current && imgRef.current) {
        // 显示所有物体的叠加效果
        const ctx = canvasRef.current.getContext('2d')
        
        // 先绘制原图
        const originalImg = new Image()
        originalImg.crossOrigin = 'anonymous'
        originalImg.onload = () => {
          ctx.clearRect(0, 0, canvasRef.current.width, canvasRef.current.height)
          ctx.drawImage(originalImg, 0, 0, canvasRef.current.width, canvasRef.current.height)
          
          // 叠加每个物体的彩色图像
          data.objects.forEach(obj => {
            if (obj.color_image) {
              const objImg = new Image()
              objImg.onload = () => {
                ctx.drawImage(objImg, 0, 0, canvasRef.current.width, canvasRef.current.height)
              }
              objImg.src = `data:image/png;base64,${obj.color_image}`
            }
          })
        }
        originalImg.src = image.url
      }
    } catch (e) {
      setColorObjects({ success: false, message: e.message })
    }
    setLoading(false)
  }, [image])

  // ── 恢复原图 ──
  const restoreOriginalImage = useCallback(() => {
    if (!originalImage || !canvasRef.current) return
    
    const ctx = canvasRef.current.getContext('2d')
    const img = new Image()
    img.crossOrigin = 'anonymous'
    img.onload = () => {
      ctx.clearRect(0, 0, canvasRef.current.width, canvasRef.current.height)
      ctx.drawImage(img, 0, 0, canvasRef.current.width, canvasRef.current.height)
    }
    img.src = originalImage.url
    
    // 清除分割结果
    setPoints([])
    setBox(null)
    setResult(null)
    setCurrentMask(null)
    setColorObjects(null)
    setSelectedColorObj(null)
  }, [originalImage])

  // ── Show single colored object on canvas ──
  const showColorObject = useCallback((obj) => {
    if (!canvasRef.current || !imgRef.current) return
    const ctx = canvasRef.current.getContext('2d')
    const img = new Image()
    img.onload = () => {
      ctx.clearRect(0, 0, canvasRef.current.width, canvasRef.current.height)
      ctx.drawImage(img, 0, 0, canvasRef.current.width, canvasRef.current.height)
    }
    img.src = `data:image/png;base64,${obj.color_image}`
    setSelectedColorObj(obj)
  }, [])

  // ── Keyboard shortcuts ──
  useEffect(() => {
    const handleKeyDown = (e) => {
      // Ignore if typing in input
      if (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA') return
      
      switch (e.key) {
        case '1':
          setTool('point')
          break
        case '2':
          setTool('box')
          break
        case 'd':
        case 'D':
          if (image && !loading) runAutoDetect()
          break
        case 's':
        case 'S':
          if (image && !loading) runAutoSegment()
          break
        case 'r':
        case 'R':
          if (image && !loading) runRecognize()
          break
        case 'c':
        case 'C':
          if (image && !loading) runExtractColors()
          break
        case 'Escape':
          clearAll()
          break
        case 'z':
        case 'Z':
          if (e.ctrlKey || e.metaKey) {
            e.preventDefault()
            setPoints(prev => prev.slice(0, -1))
          }
          break
      }
    }
    
    window.addEventListener('keydown', handleKeyDown)
    return () => window.removeEventListener('keydown', handleKeyDown)
  }, [image, loading, runAutoDetect, runAutoSegment, runRecognize, runExtractColors, clearAll])

  // ── Draw box on canvas ──
  const drawBox = box && !isDrawing ? null : null  // handled via CSS

  return (
    <div className={`app theme-${theme}`}>
      {/* Header */}
      <header className="header">
        <div className="logo">
          <div className="logo-icon">🎯</div>
          <span className="logo-text">SAM Interactive</span>
        </div>
        <div className="header-actions">
          <button 
            className="btn btn-secondary btn-sm"
            onClick={() => setTheme(theme === 'dark' ? 'light' : 'dark')}
            title={theme === 'dark' ? '切换亮色主题' : '切换暗色主题'}
          >
            {theme === 'dark' ? '☀️' : '🌙'}
          </button>
          {health && (
            <div className="status-badge">
              <span className="status-dot"></span>
              {health.device === 'cuda' ? 'GPU' : 'CPU'}
            </div>
          )}
          <button className="btn btn-secondary btn-sm" onClick={() => { clearAll(); setImage(null) }}>
            🔄 重新开始
          </button>
        </div>
      </header>

      <div className="main-content">
        {/* Toolbar */}
        <aside className="toolbar">
          {/* Mobile Toggle */}
          {isMobile && (
            <button 
              className="btn btn-secondary btn-sm"
              onClick={() => setToolbarExpanded(!toolbarExpanded)}
              style={{ 
                width: '100%', 
                marginBottom: '0.5rem',
                display: 'flex',
                justifyContent: 'space-between',
                alignItems: 'center'
              }}
            >
              <span>🔧 工具面板</span>
              <span>{toolbarExpanded ? '▲' : '▼'}</span>
            </button>
          )}

          {/* Collapsible Content */}
          <div style={{ 
            display: isMobile && !toolbarExpanded ? 'none' : 'flex',
            flexDirection: 'column',
            gap: '1rem'
          }}>
          {/* Model Selector */}
          <div className="tool-section">
            <h3>🧠 SAM 模型</h3>
            <div style={{ display: 'flex', flexDirection: 'column', gap: '0.5rem' }}>
              {availableModels.map(model => (
                <button
                  key={model.id}
                  className={`btn ${currentModel === model.id ? 'btn-primary' : 'btn-secondary'} btn-sm`}
                  onClick={() => switchSAMModel(model.id)}
                  disabled={switchingModel}
                  style={{ 
                    justifyContent: 'space-between',
                    opacity: model.exists ? 1 : 0.5
                  }}
                >
                  <span>{model.name}</span>
                  {!model.exists && <span style={{ fontSize: '0.7rem' }}>⚠️ 未下载</span>}
                  {currentModel === model.id && <span style={{ fontSize: '0.7rem' }}>✓</span>}
                </button>
              ))}
              {switchingModel && (
                <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)', textAlign: 'center' }}>
                  ⏳ 切换中...
                </div>
              )}
            </div>
          </div>

          {/* Tools */}
          <div className="tool-section">
            <h3>🔧 功能模式</h3>
            <div className="tool-grid">
              <button className={`tool-btn ${tool==='point' ? 'active' : ''}`} onClick={() => setTool('point')}>
                <span className="tool-icon">📍</span>
                <span>点击分割</span>
              </button>
              <button className={`tool-btn ${tool==='box' ? 'active' : ''}`} onClick={() => setTool('box')}>
                <span className="tool-icon">⬜</span>
                <span>框选分割</span>
              </button>
              <button className={`tool-btn ${tool==='brush' ? 'active' : ''}`} onClick={() => setTool('brush')}>
                <span className="tool-icon">🖌️</span>
                <span>画笔编辑</span>
              </button>
            </div>
          </div>

          {/* Brush settings (show when brush tool is selected) */}
          {tool === 'brush' && (
            <div className="tool-section">
              <h3>🖌️ 画笔设置</h3>
              <div style={{ display: 'flex', gap: '0.5rem', marginBottom: '0.75rem' }}>
                <button 
                  className={`btn ${brushMode === 'add' ? 'btn-primary' : 'btn-secondary'} btn-sm`}
                  onClick={() => setBrushMode('add')}
                  style={{ flex: 1 }}
                >
                  ➕ 添加
                </button>
                <button 
                  className={`btn ${brushMode === 'erase' ? 'btn-primary' : 'btn-secondary'} btn-sm`}
                  onClick={() => setBrushMode('erase')}
                  style={{ flex: 1 }}
                >
                  ➖ 擦除
                </button>
              </div>
              <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                <span style={{ fontSize: '0.8rem', color: 'var(--text-muted)' }}>大小:</span>
                <input
                  type="range"
                  min="5"
                  max="50"
                  value={brushSize}
                  onChange={(e) => setBrushSize(Number(e.target.value))}
                  style={{ flex: 1 }}
                />
                <span style={{ fontSize: '0.8rem', color: 'var(--text)' }}>{brushSize}px</span>
              </div>
            </div>
          )}

          {/* Video frame controls (show when video is loaded) */}
          {videoMode === 'video' && videoData && videoData.frames && videoData.frames.length > 0 && (
            <div className="tool-section">
              <h3>🎬 视频帧控制</h3>
              <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)', marginBottom: '0.5rem' }}>
                {videoData.filename} · {videoData.frame_count} 帧 · {videoData.info.duration?.toFixed(1)}s
              </div>
              <div style={{ display: 'flex', gap: '0.5rem', marginBottom: '0.5rem' }}>
                <button 
                  className="btn btn-secondary btn-sm"
                  onClick={() => switchFrame(Math.max(0, frameIndex - 1))}
                  disabled={frameIndex === 0}
                  style={{ flex: 1 }}
                >
                  ⏮️ 上一帧
                </button>
                <button 
                  className="btn btn-secondary btn-sm"
                  onClick={() => switchFrame(Math.min(videoData.frames.length - 1, frameIndex + 1))}
                  disabled={frameIndex === videoData.frames.length - 1}
                  style={{ flex: 1 }}
                >
                  下一帧 ⏭️
                </button>
              </div>
              <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                <span style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>帧:</span>
                <input
                  type="range"
                  min="0"
                  max={videoData.frames.length - 1}
                  value={frameIndex}
                  onChange={(e) => switchFrame(Number(e.target.value))}
                  style={{ flex: 1 }}
                />
                <span style={{ fontSize: '0.75rem', color: 'var(--text)' }}>
                  {frameIndex + 1}/{videoData.frames.length}
                </span>
              </div>
              {currentFrame && (
                <div style={{ fontSize: '0.7rem', color: 'var(--text-muted)', marginTop: '0.5rem' }}>
                  时间: {currentFrame.timestamp}s · {currentFrame.width}x{currentFrame.height}
                </div>
              )}
            </div>
          )}

          {/* Batch image controls (show when batch images are loaded) */}
          {batchImages.length > 1 && (
            <div className="tool-section">
              <h3>📂 批量图片 ({batchImages.length}张)</h3>
              <div style={{ display: 'flex', gap: '0.5rem', marginBottom: '0.5rem' }}>
                <button 
                  className="btn btn-secondary btn-sm"
                  onClick={() => switchBatchImage(Math.max(0, currentBatchIndex - 1))}
                  disabled={currentBatchIndex === 0}
                  style={{ flex: 1 }}
                >
                  ⏮️ 上一张
                </button>
                <button 
                  className="btn btn-secondary btn-sm"
                  onClick={() => switchBatchImage(Math.min(batchImages.length - 1, currentBatchIndex + 1))}
                  disabled={currentBatchIndex === batchImages.length - 1}
                  style={{ flex: 1 }}
                >
                  下一张 ⏭️
                </button>
              </div>
              <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                <span style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>图片:</span>
                <input
                  type="range"
                  min="0"
                  max={batchImages.length - 1}
                  value={currentBatchIndex}
                  onChange={(e) => switchBatchImage(Number(e.target.value))}
                  style={{ flex: 1 }}
                />
                <span style={{ fontSize: '0.75rem', color: 'var(--text)' }}>
                  {currentBatchIndex + 1}/{batchImages.length}
                </span>
              </div>
              <div style={{ fontSize: '0.7rem', color: 'var(--text-muted)', marginTop: '0.5rem' }}>
                {batchImages[currentBatchIndex]?.filename || `图片 ${currentBatchIndex + 1}`}
              </div>
            </div>
          )}

          {/* Instructions */}
          <div className="tool-section">
            <h3>📖 使用说明</h3>
            <div style={{ fontSize: '0.8rem', color: 'var(--text-muted)', lineHeight: 1.6 }}>
              {tool === 'point' ? (
                <>
                  {isMobile ? (
                    <>
                      <p>• <b>点击</b>：标记前景</p>
                      <p>• <b>长按</b>：标记背景 (待实现)</p>
                      <p>• 可添加多个点提高精度</p>
                    </>
                  ) : (
                    <>
                      <p>• <b>左键点击</b>：标记前景</p>
                      <p>• <b>Shift+点击</b>：标记背景</p>
                      <p>• 可添加多个点提高精度</p>
                    </>
                  )}
                </>
              ) : tool === 'box' ? (
                <>
                  <p>• <b>拖拽</b>：绘制选框</p>
                  <p>• 释放后自动分割</p>
                </>
              ) : (
                <>
                  <p>• <b>点击/触摸</b>：添加/擦除掩码</p>
                  <p>• 先使用点击或框选分割</p>
                  <p>• 再切换到画笔模式微调</p>
                </>
              )}
            </div>
          </div>

          {/* Points list */}
          {points.length > 0 && tool === 'point' && (
            <div className="tool-section">
              <h3>📌 标记点 ({points.length})</h3>
              <div style={{ display: 'flex', flexDirection: 'column', gap: '0.35rem' }}>
                {points.map((p, i) => (
                  <div key={i} style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', fontSize: '0.8rem' }}>
                    <span style={{ color: p.label === 1 ? 'var(--success)' : 'var(--error)' }}>
                      {p.label === 1 ? '●' : '○'}
                    </span>
                    <span>({Math.round(p.x)}, {Math.round(p.y)})</span>
                    <span style={{ color: 'var(--text-muted)', marginLeft: 'auto' }}>
                      {p.label === 1 ? '前景' : '背景'}
                    </span>
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* Actions */}
          <div className="tool-section">
            <h3>⚡ 操作</h3>
            <div className="actions-row">
              <button className="btn btn-primary" onClick={runSegmentation}
                disabled={loading || (!points.length && !box)}>
                {loading ? '⏳ 处理中...' : '✂️ 分割'}
              </button>
            </div>
            <div className="actions-row" style={{ marginTop: '0.5rem' }}>
              <button className="btn btn-secondary" onClick={runAutoDetect}
                disabled={loading || !image}>
                🔍 自动检测
              </button>
              <button className="btn btn-secondary" onClick={runAutoSegment}
                disabled={loading || !image}>
                ✨ 自动分割
              </button>
            </div>
            <div className="actions-row" style={{ marginTop: '0.5rem' }}>
              <button className="btn btn-secondary" onClick={runRecognize}
                disabled={loading || !image}>
                🏷️ 识别
              </button>
              <button className="btn btn-secondary" onClick={clearAll}>🗑️ 清除</button>
            </div>
            <div className="actions-row" style={{ marginTop: '0.5rem' }}>
              <button className="btn btn-primary" onClick={runExtractColors}
                disabled={loading || !image}
                style={{ flex: 1 }}>
                {loading ? '⏳ 提取中...' : '🎨 提取彩色物体'}
              </button>
              <button className="btn btn-secondary" onClick={restoreOriginalImage}
                disabled={!originalImage}
                style={{ flex: 1 }}>
                🖼️ 恢复原图
              </button>
            </div>
          </div>

          {/* Keyboard Shortcuts - Hide on mobile */}
          {!isMobile && (
            <div className="tool-section">
              <h3>⌨️ 快捷键</h3>
              <div style={{ 
                fontSize: '0.75rem', 
                color: 'var(--text-muted)', 
                lineHeight: 1.8,
                display: 'grid',
                gridTemplateColumns: 'auto 1fr',
                gap: '0.25rem 0.75rem'
              }}>
                <kbd style={{ 
                  background: 'var(--bg-hover)', 
                  padding: '0.1rem 0.4rem', 
                  borderRadius: '4px',
                  fontFamily: 'monospace',
                  fontSize: '0.7rem'
                }}>1</kbd>
                <span>点击分割</span>
                
                <kbd style={{ 
                  background: 'var(--bg-hover)', 
                  padding: '0.1rem 0.4rem', 
                  borderRadius: '4px',
                  fontFamily: 'monospace',
                  fontSize: '0.7rem'
                }}>2</kbd>
                <span>框选分割</span>
                
                <kbd style={{ 
                  background: 'var(--bg-hover)', 
                  padding: '0.1rem 0.4rem', 
                  borderRadius: '4px',
                  fontFamily: 'monospace',
                  fontSize: '0.7rem'
                }}>D</kbd>
                <span>自动检测</span>
                
                <kbd style={{ 
                  background: 'var(--bg-hover)', 
                  padding: '0.1rem 0.4rem', 
                  borderRadius: '4px',
                  fontFamily: 'monospace',
                  fontSize: '0.7rem'
                }}>S</kbd>
                <span>自动分割</span>
                
                <kbd style={{ 
                  background: 'var(--bg-hover)', 
                  padding: '0.1rem 0.4rem', 
                  borderRadius: '4px',
                  fontFamily: 'monospace',
                  fontSize: '0.7rem'
                }}>R</kbd>
                <span>图像识别</span>
                
                <kbd style={{ 
                  background: 'var(--bg-hover)', 
                  padding: '0.1rem 0.4rem', 
                  borderRadius: '4px',
                  fontFamily: 'monospace',
                  fontSize: '0.7rem'
                }}>C</kbd>
                <span>彩色提取</span>
                
                <kbd style={{ 
                  background: 'var(--bg-hover)', 
                  padding: '0.1rem 0.4rem', 
                  borderRadius: '4px',
                  fontFamily: 'monospace',
                  fontSize: '0.7rem'
                }}>Esc</kbd>
                <span>清除所有</span>
                
                <kbd style={{ 
                  background: 'var(--bg-hover)', 
                  padding: '0.1rem 0.4rem', 
                  borderRadius: '4px',
                  fontFamily: 'monospace',
                  fontSize: '0.7rem'
                }}>Ctrl+Z</kbd>
                <span>撤销标记</span>
              </div>
            </div>
          )}
          </div> {/* End collapsible content */}
        </aside>

        {/* Canvas Area */}
        <main className="canvas-area">
          {!image ? (
            <div
              className={`upload-zone ${dragOver ? 'drag-over' : ''}`}
              onDragOver={(e) => { e.preventDefault(); setDragOver(true) }}
              onDragLeave={() => setDragOver(false)}
              onDrop={handleDrop}
            >
              <div className="upload-icon">🖼️</div>
              <div className="upload-title">上传图片或视频开始分割</div>
              <div className="upload-subtitle">拖拽文件到此处，或点击选择</div>
              <div className="upload-subtitle">支持 JPG, PNG, WebP, MP4, AVI 格式</div>
              
              <div style={{ display: 'flex', gap: '0.75rem', marginTop: '1rem', flexWrap: 'wrap', justifyContent: 'center' }}>
                <button 
                  className="btn btn-primary"
                  onClick={(e) => { e.stopPropagation(); fileInputRef.current?.click() }}
                >
                  📁 选择图片
                </button>
                <button 
                  className="btn btn-secondary"
                  onClick={(e) => { 
                    e.stopPropagation()
                    const input = document.createElement('input')
                    input.type = 'file'
                    input.multiple = true
                    input.accept = 'image/*'
                    input.onchange = (ev) => handleBatchUpload(ev.target.files)
                    input.click()
                  }}
                >
                  📂 批量图片
                </button>
                <button 
                  className="btn btn-secondary"
                  onClick={(e) => { 
                    e.stopPropagation()
                    const input = document.createElement('input')
                    input.type = 'file'
                    input.accept = 'video/*'
                    input.onchange = (ev) => handleVideoUpload(ev.target.files[0])
                    input.click()
                  }}
                >
                  🎬 视频文件
                </button>
                {cameras.length > 0 && (
                  <button 
                    className="btn btn-secondary"
                    onClick={(e) => { 
                      e.stopPropagation()
                      startCameraStream(0)
                    }}
                  >
                    📷 摄像头
                  </button>
                )}
              </div>
              
              <input
                ref={fileInputRef}
                type="file"
                accept="image/*"
                style={{ display: 'none' }}
                onChange={(e) => handleUpload(e.target.files[0])}
              />
            </div>
          ) : videoMode === 'camera' && cameraStreamUrl ? (
            <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: '1rem' }}>
              <div style={{ position: 'relative' }}>
                <img 
                  src={cameraStreamUrl} 
                  alt="Camera Stream"
                  style={{ 
                    maxWidth: '100%', 
                    maxHeight: '70vh',
                    borderRadius: '8px',
                    border: '1px solid var(--border)'
                  }}
                />
                <button
                  className="btn btn-secondary btn-sm"
                  onClick={stopCameraStream}
                  style={{ position: 'absolute', top: '10px', right: '10px' }}
                >
                  ⏹️ 停止
                </button>
              </div>
              <div style={{ fontSize: '0.8rem', color: 'var(--text-muted)' }}>
                摄像头 #{selectedCamera} · 实时流
              </div>
            </div>
          ) : (
            <div className="canvas-container">
              <canvas
                ref={canvasRef}
                onClick={handleCanvasClick}
                onMouseDown={handleMouseDown}
                onMouseMove={handleMouseMove}
                onMouseUp={handleMouseUp}
                onMouseLeave={handleMouseUp}
                onTouchStart={handleTouchStart}
                onTouchMove={handleTouchMove}
                onTouchEnd={handleTouchEnd}
                style={{
                  cursor: tool === 'point' ? 'crosshair' : (isDrawing ? 'grabbing' : 'crosshair'),
                  touchAction: 'none'  // Prevent default touch behaviors
                }}
              />
              <canvas ref={overlayRef} className="mask-overlay" />

              {/* Point markers */}
              {points.map((p, i) => imgRef.current && (
                <div
                  key={i}
                  className={`point-marker ${p.label === 1 ? 'foreground' : 'background'}`}
                  style={{
                    left: p.dx,
                    top: p.dy,
                  }}
                />
              ))}

              {/* Box overlay */}
              {box && (
                <div className="box-overlay" style={{
                  left: Math.min(box.x1, box.x2),
                  top: Math.min(box.y1, box.y2),
                  width: Math.abs(box.x2 - box.x1),
                  height: Math.abs(box.y2 - box.y1),
                }} />
              )}

              {/* Loading */}
              {loading && (
                <div className="loading-overlay">
                  <div className="loading-spinner"></div>
                  <span>SAM 正在分析...</span>
                </div>
              )}
            </div>
          )}
        </main>

        {/* Results Panel */}
        <aside className="results-panel">
          <div className="tool-section">
            <h3>📊 分割结果</h3>
            {!result ? (
              <div className="empty-state">
                <span className="icon">🎯</span>
                <p>在图片上标记点或框选区域，然后点击"开始分割"</p>
              </div>
            ) : result.success ? (
              <>
                <div className="result-metrics">
                  <div className="metric-item">
                    <span className="metric-label">置信度</span>
                    <span className="metric-value">{(result.score * 100).toFixed(1)}%</span>
                  </div>
                  <div className="metric-item">
                    <span className="metric-label">区域面积</span>
                    <span className="metric-value">{result.area?.toLocaleString() || 'N/A'} px</span>
                  </div>
                  {result.bbox && (
                    <>
                      <div className="metric-item">
                        <span className="metric-label">边界框</span>
                        <span className="metric-value" style={{ fontSize: '0.75rem' }}>
                          {result.bbox.map(v => Math.round(v)).join(', ')}
                        </span>
                      </div>
                      <div className="metric-item">
                        <span className="metric-label">尺寸</span>
                        <span className="metric-value">
                          {Math.round(result.bbox[2]-result.bbox[0])} × {Math.round(result.bbox[3]-result.bbox[1])}
                        </span>
                      </div>
                    </>
                  )}
                </div>
                {result.overlay && (
                  <div className="mask-preview">
                    <h4 style={{ fontSize: '0.8rem', marginBottom: '0.5rem' }}>🎨 分割效果</h4>
                    <img src={`data:image/png;base64,${result.overlay}`} alt="Overlay" />
                  </div>
                )}
                {result.mask && (
                  <div className="mask-preview" style={{ marginTop: '0.75rem' }}>
                    <h4 style={{ fontSize: '0.8rem', marginBottom: '0.5rem' }}>🎭 掩码</h4>
                    <img src={`data:image/png;base64,${result.mask}`} alt="Mask" />
                  </div>
                )}
              </>
            ) : (
              <div className="empty-state">
                <span className="icon">❌</span>
                <p>{result.message || '分割失败'}</p>
              </div>
            )}
          </div>

          {/* Export */}
          {result?.success && (
            <div className="tool-section">
              <h3>💾 导出</h3>
              <div className="actions-row" style={{ flexDirection: 'column', gap: '0.5rem' }}>
                <button className="btn btn-secondary btn-sm" onClick={() => {
                  const a = document.createElement('a')
                  a.href = `data:image/png;base64,${result.overlay}`
                  a.download = 'sam_overlay.png'
                  a.click()
                }}>
                  🎨 下载效果图
                </button>
                <button className="btn btn-secondary btn-sm" onClick={() => {
                  const a = document.createElement('a')
                  a.href = `data:image/png;base64,${result.mask}`
                  a.download = 'sam_mask.png'
                  a.click()
                }}>
                  🎭 下载掩码
                </button>
                <button className="btn btn-secondary btn-sm" onClick={() => {
                  const a = document.createElement('a')
                  a.href = `${API}/api/export/coco/${image.id}`
                  a.download = `sam_coco_${image.id}.json`
                  a.click()
                }}>
                  📦 导出 COCO 格式
                </button>
                <button className="btn btn-secondary btn-sm" onClick={() => {
                  const a = document.createElement('a')
                  a.href = `${API}/api/export/yolo/${image.id}`
                  a.download = `sam_yolo_${image.id}.zip`
                  a.click()
                }}>
                  📦 导出 YOLO 格式
                </button>
              </div>
            </div>
          )}

          {/* Auto Detect Results */}
          {detectResult?.success && (
            <div className="tool-section">
              <h3>🔍 自动检测结果 ({detectResult.count} 个区域)</h3>
              {detectResult.overlay && (
                <div className="mask-preview" style={{ marginBottom: '0.75rem' }}>
                  <img src={`data:image/png;base64,${detectResult.overlay}`} alt="检测效果图" />
                </div>
              )}
              <div style={{ maxHeight: '200px', overflowY: 'auto', fontSize: '0.8rem' }}>
                {detectResult.detections.map((d, i) => (
                  <div key={i} style={{
                    display: 'flex', justifyContent: 'space-between',
                    padding: '0.4rem 0.5rem', borderBottom: '1px solid var(--border)',
                    cursor: d.mask ? 'pointer' : 'default',
                    borderRadius: '4px',
                    transition: 'background 0.2s'
                  }}
                    onClick={() => {
                      if (d.mask) {
                        // 在画布上显示单个物体的掩码
                        const ctx = canvasRef.current.getContext('2d')
                        const img = new Image()
                        img.onload = () => {
                          ctx.clearRect(0, 0, canvasRef.current.width, canvasRef.current.height)
                          // 先画原图
                          const origImg = imgRef.current.el
                          ctx.drawImage(origImg, 0, 0, canvasRef.current.width, canvasRef.current.height)
                          // 叠加半透明掩码
                          ctx.globalAlpha = 0.5
                          ctx.drawImage(img, 0, 0, canvasRef.current.width, canvasRef.current.height)
                          ctx.globalAlpha = 1.0
                        }
                        img.src = `data:image/png;base64,${d.mask}`
                      }
                    }}
                    onMouseEnter={(e) => e.target.style.background = 'rgba(99,102,241,0.1)'}
                    onMouseLeave={(e) => e.target.style.background = 'transparent'}
                  >
                    <span style={{ 
                      display: 'flex', alignItems: 'center', gap: '0.4rem',
                      fontWeight: i === 0 ? 600 : 400,
                      color: i === 0 ? 'var(--primary)' : 'inherit'
                    }}>
                      <span style={{
                        width: '10px', height: '10px', borderRadius: '2px',
                        background: `hsl(${d.id * 25}, 70%, 50%)`
                      }}></span>
                      {i === 0 ? '🏆' : ''} {d.label}
                    </span>
                    <span style={{ color: 'var(--text-muted)' }}>
                      {d.confidence ? `${(d.confidence * 100).toFixed(0)}%` : `${(d.score * 100).toFixed(0)}%`}
                    </span>
                  </div>
                ))}
              </div>
              <div style={{ fontSize: '0.7rem', color: 'var(--text-muted)', marginTop: '0.5rem', textAlign: 'center' }}>
                💡 点击物体名称查看其分割掩码
              </div>
            </div>
          )}

          {/* Auto Segment Results */}
          {autoSegResult?.success && (
            <div className="tool-section">
              <h3>✨ 自动分割结果 ({autoSegResult.count} 个物体)</h3>
              {autoSegResult.overlay && (
                <div className="mask-preview" style={{ marginBottom: '0.75rem' }}>
                  <img src={`data:image/png;base64,${autoSegResult.overlay}`} alt="分割效果图" />
                </div>
              )}
              <div className="result-metrics">
                <div className="metric-item">
                  <span className="metric-label">检测物体数</span>
                  <span className="metric-value">{autoSegResult.count}</span>
                </div>
                <div className="metric-item">
                  <span className="metric-label">总覆盖面积</span>
                  <span className="metric-value">
                    {autoSegResult.detections?.reduce((sum, d) => sum + d.area, 0)?.toLocaleString() || 0} px
                  </span>
                </div>
              </div>
              <div style={{ maxHeight: '150px', overflowY: 'auto', fontSize: '0.8rem', marginTop: '0.5rem' }}>
                {autoSegResult.detections?.map((d, i) => (
                  <div key={i} style={{
                    display: 'flex', justifyContent: 'space-between',
                    padding: '0.4rem 0.5rem', borderBottom: '1px solid var(--border)',
                    cursor: d.mask ? 'pointer' : 'default',
                    borderRadius: '4px',
                    transition: 'background 0.2s'
                  }}
                    onClick={() => {
                      if (d.mask) {
                        const ctx = canvasRef.current.getContext('2d')
                        const img = new Image()
                        img.onload = () => {
                          ctx.clearRect(0, 0, canvasRef.current.width, canvasRef.current.height)
                          const origImg = imgRef.current.el
                          ctx.drawImage(origImg, 0, 0, canvasRef.current.width, canvasRef.current.height)
                          ctx.globalAlpha = 0.5
                          ctx.drawImage(img, 0, 0, canvasRef.current.width, canvasRef.current.height)
                          ctx.globalAlpha = 1.0
                        }
                        img.src = `data:image/png;base64,${d.mask}`
                      }
                    }}
                    onMouseEnter={(e) => e.target.style.background = 'rgba(99,102,241,0.1)'}
                    onMouseLeave={(e) => e.target.style.background = 'transparent'}
                  >
                    <span style={{ display: 'flex', alignItems: 'center', gap: '0.3rem' }}>
                      <span style={{
                        width: '10px', height: '10px', borderRadius: '2px',
                        background: `hsl(${d.id * 25}, 70%, 50%)`
                      }}></span>
                      {d.label}
                    </span>
                    <span style={{ color: 'var(--text-muted)' }}>
                      {d.confidence ? `${(d.confidence * 100).toFixed(0)}%` : `${(d.score * 100).toFixed(0)}%`}
                    </span>
                  </div>
                ))}
              </div>
              <div style={{ fontSize: '0.7rem', color: 'var(--text-muted)', marginTop: '0.5rem', textAlign: 'center' }}>
                💡 点击物体名称查看其分割掩码
              </div>
            </div>
          )}

          {/* Recognize Results */}
          {recognizeResult?.success && (
            <div className="tool-section">
              <h3>🏷️ 图像识别</h3>

              {/* 物体识别结果 */}
              {recognizeResult.classifications && (
                <div style={{ marginBottom: '0.75rem' }}>
                  <span className="metric-label">🎯 识别物体</span>
                  <div style={{ marginTop: '0.3rem' }}>
                    {recognizeResult.classifications.map((c, i) => (
                      <div key={i} style={{
                        display: 'flex', justifyContent: 'space-between', alignItems: 'center',
                        padding: '0.4rem 0', borderBottom: '1px solid var(--border)',
                        fontSize: '0.85rem'
                      }}>
                        <span style={{
                          fontWeight: i === 0 ? 600 : 400,
                          color: i === 0 ? 'var(--primary)' : 'inherit'
                        }}>
                          {i === 0 ? '🏆' : `${i + 1}.`} {c.label}
                        </span>
                        <span style={{
                          background: `rgba(99,102,241,${c.prob})`,
                          padding: '0.15rem 0.5rem', borderRadius: '4px', fontSize: '0.75rem',
                          color: 'white'
                        }}>
                          {(c.prob * 100).toFixed(1)}%
                        </span>
                      </div>
                    ))}
                  </div>
                </div>
              )}

              <div className="result-metrics">
                <div className="metric-item">
                  <span className="metric-label">场景</span>
                  <span className="metric-value" style={{ fontSize: '0.8rem' }}>
                    {recognizeResult.scene}
                  </span>
                </div>
                <div className="metric-item">
                  <span className="metric-label">亮度</span>
                  <span className="metric-value">{recognizeResult.brightness}</span>
                </div>
                <div className="metric-item">
                  <span className="metric-label">对比度</span>
                  <span className="metric-value">{recognizeResult.contrast}</span>
                </div>
                <div className="metric-item">
                  <span className="metric-label">尺寸</span>
                  <span className="metric-value">
                    {recognizeResult.image_size[0]} × {recognizeResult.image_size[1]}
                  </span>
                </div>
              </div>
              <div style={{ marginTop: '0.75rem' }}>
                <span className="metric-label">主色调</span>
                <div style={{ display: 'flex', gap: '4px', marginTop: '0.3rem' }}>
                  {recognizeResult.dominant_colors.slice(0, 5).map((c, i) => (
                    <div key={i} style={{
                      width: '24px', height: '24px', borderRadius: '4px',
                      background: `rgb(${c.rgb.join(',')})`,
                      border: '1px solid var(--border)',
                      title: `${(c.ratio * 100).toFixed(1)}%`
                    }} title={`${(c.ratio * 100).toFixed(1)}%`} />
                  ))}
                </div>
              </div>
            </div>
          )}

          {/* Color Objects Results */}
          {colorObjects?.success && (
            <div className="tool-section">
              <h3>🎨 彩色物体提取 ({colorObjects.count} 个)</h3>
              
              {/* 提示信息 */}
              <div style={{ 
                fontSize: '0.75rem', 
                color: 'var(--text-muted)', 
                marginBottom: '0.5rem',
                padding: '0.5rem',
                background: 'rgba(99,102,241,0.1)',
                borderRadius: '6px'
              }}>
                💡 点击物体查看效果 · 置信度 ≥ 60%
              </div>
              
              {/* 彩色物体网格 */}
              <div style={{
                display: 'grid',
                gridTemplateColumns: 'repeat(3, 1fr)',
                gap: '0.5rem',
                maxHeight: '300px',
                overflowY: 'auto',
                padding: '0.25rem'
              }}>
                {colorObjects.objects.map((obj, i) => (
                  <div
                    key={i}
                    onClick={() => showColorObject(obj)}
                    style={{
                      cursor: 'pointer',
                      border: selectedColorObj?.id === obj.id ? '2px solid var(--primary)' : '1px solid var(--border)',
                      borderRadius: '8px',
                      padding: '0.25rem',
                      background: 'var(--bg-tertiary)',
                      transition: 'all 0.2s'
                    }}
                    onMouseEnter={(e) => e.currentTarget.style.transform = 'scale(1.05)'}
                    onMouseLeave={(e) => e.currentTarget.style.transform = 'scale(1)'}
                  >
                    <img
                      src={`data:image/png;base64,${obj.color_image}`}
                      alt={obj.label}
                      style={{
                        width: '100%',
                        height: '60px',
                        objectFit: 'contain',
                        borderRadius: '4px',
                        background: 'repeating-conic-gradient(#ddd 0% 25%, white 0% 50%) 50% / 10px 10px'
                      }}
                    />
                    <div style={{
                      fontSize: '0.65rem',
                      textAlign: 'center',
                      marginTop: '0.25rem',
                      color: 'var(--text-secondary)',
                      overflow: 'hidden',
                      textOverflow: 'ellipsis',
                      whiteSpace: 'nowrap'
                    }}>
                      {obj.label}
                    </div>
                    <div style={{
                      fontSize: '0.6rem',
                      textAlign: 'center',
                      color: obj.confidence >= 0.8 ? 'var(--success)' : 
                             obj.confidence >= 0.6 ? 'var(--warning)' : 'var(--error)',
                      fontWeight: 600
                    }}>
                      {(obj.confidence * 100).toFixed(0)}%
                    </div>
                  </div>
                ))}
              </div>

              {/* 恢复原图按钮 */}
              <button 
                className="btn btn-secondary btn-sm"
                onClick={restoreOriginalImage}
                style={{ width: '100%', marginTop: '0.75rem' }}
              >
                🖼️ 恢复原图
              </button>

              {/* 选中物体详情 */}
              {selectedColorObj && (
                <div style={{
                  marginTop: '0.75rem',
                  padding: '0.75rem',
                  background: 'var(--bg-tertiary)',
                  borderRadius: '8px',
                  border: '1px solid var(--border)'
                }}>
                  <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', marginBottom: '0.5rem' }}>
                    <img
                      src={`data:image/png;base64,${selectedColorObj.color_image}`}
                      alt={selectedColorObj.label}
                      style={{
                        width: '60px',
                        height: '60px',
                        objectFit: 'contain',
                        borderRadius: '4px',
                        background: 'repeating-conic-gradient(#ddd 0% 25%, white 0% 50%) 50% / 10px 10px'
                      }}
                    />
                    <div>
                      <div style={{ fontWeight: 600, color: 'var(--primary)' }}>
                        {selectedColorObj.label}
                      </div>
                      <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>
                        置信度: {selectedColorObj.confidence ? `${(selectedColorObj.confidence * 100).toFixed(1)}%` : `${(selectedColorObj.score * 100).toFixed(1)}%`}
                      </div>
                      <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>
                        面积: {selectedColorObj.area?.toLocaleString()} px
                      </div>
                    </div>
                  </div>
                  <button
                    className="btn btn-secondary btn-sm"
                    onClick={() => {
                      const a = document.createElement('a')
                      a.href = `data:image/png;base64,${selectedColorObj.color_image}`
                      a.download = `${selectedColorObj.label}.png`
                      a.click()
                    }}
                    style={{ width: '100%' }}
                  >
                    💾 下载彩色物体
                  </button>
                </div>
              )}

              <div style={{ fontSize: '0.7rem', color: 'var(--text-muted)', marginTop: '0.5rem', textAlign: 'center' }}>
                💡 点击物体缩略图查看详情并下载
              </div>
            </div>
          )}

          {/* History Panel */}
          <div className="tool-section">
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '0.5rem' }}>
              <h3 style={{ margin: 0 }}>📋 分割历史 ({history.length})</h3>
              {history.length > 0 && (
                <div style={{ display: 'flex', gap: '0.25rem' }}>
                  <button 
                    className="btn btn-secondary btn-sm"
                    onClick={() => exportHistory('json')}
                    style={{ padding: '0.25rem 0.5rem', fontSize: '0.7rem' }}
                    title="导出 JSON"
                  >
                    📄 JSON
                  </button>
                  <button 
                    className="btn btn-secondary btn-sm"
                    onClick={() => exportHistory('csv')}
                    style={{ padding: '0.25rem 0.5rem', fontSize: '0.7rem' }}
                    title="导出 CSV"
                  >
                    📊 CSV
                  </button>
                  <button 
                    className="btn btn-secondary btn-sm"
                    onClick={clearHistory}
                    style={{ padding: '0.25rem 0.5rem', fontSize: '0.7rem' }}
                    title="清空历史"
                  >
                    🗑️
                  </button>
                </div>
              )}
            </div>
            
            {history.length === 0 ? (
              <div className="empty-state" style={{ padding: '1rem' }}>
                <p style={{ fontSize: '0.8rem' }}>暂无历史记录</p>
              </div>
            ) : (
              <div style={{ maxHeight: '250px', overflowY: 'auto' }}>
                {[...history].reverse().map((entry) => (
                  <div
                    key={entry.id}
                    onClick={() => loadHistoryEntry(entry.id)}
                    style={{
                      display: 'flex',
                      alignItems: 'center',
                      justifyContent: 'space-between',
                      padding: '0.5rem',
                      marginBottom: '0.25rem',
                      borderRadius: '6px',
                      cursor: 'pointer',
                      background: selectedHistoryId === entry.id ? 'rgba(99,102,241,0.15)' : 'var(--bg-card)',
                      border: selectedHistoryId === entry.id ? '1px solid var(--primary)' : '1px solid var(--border)',
                      transition: 'all 0.2s'
                    }}
                    onMouseEnter={(e) => {
                      if (selectedHistoryId !== entry.id) {
                        e.currentTarget.style.background = 'var(--bg-hover)'
                      }
                    }}
                    onMouseLeave={(e) => {
                      if (selectedHistoryId !== entry.id) {
                        e.currentTarget.style.background = 'var(--bg-card)'
                      }
                    }}
                  >
                    <div style={{ flex: 1 }}>
                      <div style={{ 
                        display: 'flex', 
                        alignItems: 'center', 
                        gap: '0.5rem',
                        fontSize: '0.85rem',
                        fontWeight: selectedHistoryId === entry.id ? 600 : 400
                      }}>
                        <span>{entry.tool === 'point' ? '📍' : '⬜'}</span>
                        <span style={{ 
                          color: selectedHistoryId === entry.id ? 'var(--primary)' : 'var(--text)' 
                        }}>
                          {entry.label || '分割区域'}
                        </span>
                      </div>
                      <div style={{ 
                        fontSize: '0.7rem', 
                        color: 'var(--text-muted)',
                        marginTop: '0.2rem',
                        display: 'flex',
                        gap: '0.75rem'
                      }}>
                        <span>置信度: {(entry.score * 100).toFixed(1)}%</span>
                        <span>面积: {entry.area?.toLocaleString()} px</span>
                      </div>
                    </div>
                    <button
                      onClick={(e) => {
                        e.stopPropagation()
                        deleteHistoryEntry(entry.id)
                      }}
                      style={{
                        background: 'none',
                        border: 'none',
                        cursor: 'pointer',
                        color: 'var(--text-muted)',
                        padding: '0.25rem',
                        borderRadius: '4px',
                        fontSize: '0.8rem'
                      }}
                      onMouseEnter={(e) => e.target.style.color = 'var(--error)'}
                      onMouseLeave={(e) => e.target.style.color = 'var(--text-muted)'}
                    >
                      ❌
                    </button>
                  </div>
                ))}
              </div>
            )}
            
            <div style={{ fontSize: '0.7rem', color: 'var(--text-muted)', marginTop: '0.5rem', textAlign: 'center' }}>
              💡 点击历史记录恢复分割结果
            </div>
          </div>
        </aside>
      </div>

      {/* Info Bar */}
      <footer className="info-bar">
        <span className="info-item">🖼️ {image ? `${image.width} × ${image.height}` : '未加载图片'}</span>
        <span className="info-item">🔧 {tool === 'point' ? '点击分割' : '框选分割'}</span>
        <span className="info-item">📌 {points.length} 个标记点</span>
        {health && <span className="info-item">💻 {health.device}</span>}
      </footer>
    </div>
  )
}

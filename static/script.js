async function startTranscription() {
    try {
        const response = await fetch('/start', { method: 'POST' });
        const data = await response.json();
        updateStatus(data);
        if (data.status === 'success') {
            document.getElementById('startBtn').disabled = true;
            document.getElementById('stopBtn').disabled = false;
        }
    } catch (error) {
        console.error('Error:', error);
        updateStatus({ status: 'error', message: '启动失败: ' + error.message });
    }
}

async function stopTranscription() {
    try {
        const response = await fetch('/stop', { method: 'POST' });
        const data = await response.json();
        updateStatus(data);
        if (data.status === 'success') {
            document.getElementById('startBtn').disabled = false;
            document.getElementById('stopBtn').disabled = true;
        }
    } catch (error) {
        console.error('Error:', error);
        updateStatus({ status: 'error', message: '停止失败: ' + error.message });
    }
}

function updateStatus(data) {
    const statusDiv = document.getElementById('status');
    statusDiv.textContent = data.message;
    statusDiv.className = 'status ' + data.status;
}

// 定期检查状态
async function checkStatus() {
    try {
        const response = await fetch('/status');
        const data = await response.json();
        document.getElementById('startBtn').disabled = data.status === 'running';
        document.getElementById('stopBtn').disabled = data.status === 'stopped';
        updateStatus(data);
    } catch (error) {
        console.error('Error:', error);
    }
}

// 页面加载时检查状态
window.onload = function() {
    checkStatus();
    // 每5秒检查一次状态
    setInterval(checkStatus, 5000);
};
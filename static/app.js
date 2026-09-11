const API_BASE = '/api/v1';

document.addEventListener('DOMContentLoaded', function() {
    const genForm = document.getElementById('generation-form');
    const optForm = document.getElementById('optimization-form');

    if (genForm) {
        genForm.addEventListener('submit', handleGeneration);
    }

    if (optForm) {
        optForm.addEventListener('submit', handleOptimization);
    }
});

async function handleGeneration(e) {
    e.preventDefault();
    const form = e.target;
    const resultArea = document.getElementById('generation-result');

    const formData = new FormData();
    formData.append('questionnaire', form.questionnaire.files[0]);
    formData.append('indicators', form.indicators.files[0]);

    if (form.expert_judgments.files[0]) {
        formData.append('expert_judgments', form.expert_judgments.files[0]);
    }
    if (form.expert_authority.files[0]) {
        formData.append('expert_authority', form.expert_authority.files[0]);
    }
    if (form.vehicle_scores.files[0]) {
        formData.append('vehicle_scores', form.vehicle_scores.files[0]);
    }

    try {
        resultArea.innerHTML = '<p>处理中...</p>';
        resultArea.classList.add('show');

        const response = await fetch(`${API_BASE}/runs/generation`, {
            method: 'POST',
            body: formData,
        });

        const data = await response.json();

        if (response.ok) {
            resultArea.innerHTML = `
                <p style="color: green;">生成成功!</p>
                <p>Run ID: ${data.run_id}</p>
                <p>状态: ${data.status}</p>
            `;
        } else {
            resultArea.innerHTML = `<p style="color: red;">错误: ${data.detail}</p>`;
        }
    } catch (error) {
        resultArea.innerHTML = `<p style="color: red;">请求失败: ${error.message}</p>`;
    }
}

async function handleOptimization(e) {
    e.preventDefault();
    const form = e.target;
    const resultArea = document.getElementById('optimization-result');

    const formData = new FormData();
    formData.append('base_run_id', form.base_run_id.value);
    formData.append('optimization_samples', form.optimization_samples.files[0]);

    try {
        resultArea.innerHTML = '<p>处理中...</p>';
        resultArea.classList.add('show');

        const response = await fetch(`${API_BASE}/runs/${form.base_run_id.value}/optimization`, {
            method: 'POST',
            body: formData,
        });

        const data = await response.json();

        if (response.ok) {
            resultArea.innerHTML = `
                <p style="color: green;">优化成功!</p>
                <p>Train MAE: ${data.train_mae?.toFixed(4) || 'N/A'}</p>
                <p>Validation MAE: ${data.val_mae?.toFixed(4) || 'N/A'}</p>
                <p>Test MAE: ${data.test_mae?.toFixed(4) || 'N/A'}</p>
            `;
        } else {
            resultArea.innerHTML = `<p style="color: red;">错误: ${data.detail}</p>`;
        }
    } catch (error) {
        resultArea.innerHTML = `<p style="color: red;">请求失败: ${error.message}</p>`;
    }
}

// script.js (Versi Baru)

document.addEventListener('DOMContentLoaded', () => {
    
    const checkButton = document.getElementById('checkButton');
    const newsTitleInput = document.getElementById('newsTitle');
    const resultContainer = document.getElementById('resultContainer'); // Kita pakai container

    checkButton.addEventListener('click', () => {
        const titleToPredict = newsTitleInput.value;

        if (titleToPredict.trim() === '') {
            resultContainer.innerHTML = '<p id="resultArea">Harap masukkan judul berita terlebih dahulu.</p>';
            return;
        }

        // Tampilkan spinner
        resultContainer.innerHTML = '<div class="spinner"></div>';

        fetch('http://127.0.0.1:8000/predict', {
            /* ... (sisa kode fetch tetap sama) ... */
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ title: titleToPredict }),
        })
        .then(response => response.json())
        .then(data => {
            // Tampilkan hasil dengan efek animasi
            resultContainer.innerHTML = `<p id="resultArea">Hasil: ${data.prediction}</p>`;
        })
        .catch(error => {
            console.error('Error:', error);
            resultContainer.innerHTML = '<p id="resultArea">Terjadi kesalahan. Pastikan server API Anda sudah berjalan.</p>';
        });
    });
});
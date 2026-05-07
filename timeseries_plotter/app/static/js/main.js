document.getElementById('upload-form').onsubmit = function(event) {
    event.preventDefault();
    const formData = new FormData(this);
    
    fetch('/upload', {
        method: 'POST',
        body: formData
    })
    .then(response => response.json())
    .then(data => {
        if (data.success) {
            // Redirect to the visualization page with the selected table
            window.location.href = `/visualize?table=${data.table}`;
        } else {
            alert('Error uploading file: ' + data.message);
        }
    })
    .catch(error => {
        console.error('Error:', error);
    });
};

document.getElementById('table-select').onchange = function() {
    const selectedTable = this.value;
    // Additional logic can be added here if needed
};
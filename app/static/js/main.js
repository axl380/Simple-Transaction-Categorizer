document.getElementById('transactionForm').addEventListener('submit', function(event) {
    event.preventDefault();

    const merchant = event.target.merchant.value.trim();
    let amount = event.target.amount.value.trim();
    const notes = event.target.notes.value.trim(); 

    // Clean and format amount
    amount = amount.replace(/[$,\s;]/g, '');

    if (isNaN(amount) || amount === '') {
        alert('Invalid amount!');
        return;
    }

    const formattedAmount = parseFloat(amount).toFixed(2);

    fetch('/', {
        method: 'POST',
        body: new URLSearchParams({ merchant: merchant })
    })
    .then(response => response.json())
    .then(data => {
        const transactionId = Date.now();
        const category = data.category;
        const confidence = (data.confidence * 100).toFixed(2); 

        // Add transaction to the table
        const transactionsTable = document.getElementById('transactionsTable').getElementsByTagName('tbody')[0];
        const row = transactionsTable.insertRow();

        // Generate timestamp
        const now = new Date();
        const isoTimestamp = now.toISOString(); // CSV Export
        const localTimestamp = now.toLocaleString(); // UI Display

        row.insertCell(0).textContent = transactionId;

        // Insert timestamp cell, display as local, store as ISO
        const timestampcell = row.insertCell(1);
        timestampcell.textContent = localTimestamp;
        timestampcell.setAttribute('data-iso', isoTimestamp);

        // Insert remaining data
        row.insertCell(2).textContent = merchant;
        row.insertCell(3).textContent = `$${formattedAmount}`;
        row.insertCell(4).textContent = category;
        row.insertCell(5).textContent = confidence + '%';
        row.insertCell(6).textContent = notes;

        // Clear form
        event.target.reset();
    });
});

// Export to CSV
document.getElementById('exportCSV').addEventListener('click', function () {
    const rows = document.querySelectorAll('table tr');

    if (rows.length === 1) {
        alert('No transactions to export.');
        return;
    }

    const csv = Array.from(rows).map(row =>
        Array.from(row.cells).map((cell, index) => {
            if (index === 1) {  // Timestamp column
                return `"${cell.getAttribute('data-iso')}"`;  // Use ISO timestamp for CSV
            } else
            if (index === 3) {  // Amount column
                return `"${cell.textContent.replace('$', '')}"`;  // Remove $ for CSV
            }
            return `"${cell.textContent}"`;
        }).join(',')
    ).join('\n');    

    const blob = new Blob([csv], { type: 'text/csv' });
    const url = window.URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'transactions.csv';
    a.click();
});

// Dark Mode Toggle with Persistence
const darkModeToggle = document.getElementById('darkModeToggle');
const icon = darkModeToggle.querySelector('.material-icons');

// Check for saved dark mode preference on load
if (localStorage.getItem('darkMode') === 'true') {
    document.body.classList.add('dark-mode');
    icon.textContent = 'light_mode';
} else {
    icon.textContent = 'dark_mode'; 
}

// Toggle dark mode on button click
darkModeToggle.addEventListener('click', function () {
    document.body.classList.toggle('dark-mode');

    // Update icon based on current mode
    const isDarkMode = document.body.classList.contains('dark-mode');
    icon.textContent = isDarkMode ? 'light_mode' : 'dark_mode';

    // Save preference to localStorage
    localStorage.setItem('darkMode', isDarkMode);
});

// Confirmation Prompt on Refresh if Transactions Exist
window.addEventListener('beforeunload', function(e) {
    const transactionsExist = document.querySelectorAll('#transactionsTable tbody tr').length > 0;
    if (transactionsExist) {
        e.preventDefault();
    }
});

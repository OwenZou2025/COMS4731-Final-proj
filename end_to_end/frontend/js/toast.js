// Toast Notification System
const toast = {
    container: null,
    
    init() {
        this.container = document.getElementById('toastContainer');
    },
    
    show(options) {
        const { title, message, duration = 8000 } = options;
        
        // Dismiss any existing toasts first (only one toast at a time)
        this.dismissAll();
        
        const toastEl = document.createElement('div');
        toastEl.className = 'toast';
        toastEl.innerHTML = `
            <div class="toast-content">
                <div class="toast-title">${title}</div>
                ${message ? `<div class="toast-message">${message}</div>` : ''}
            </div>
        `;
        
        this.container.appendChild(toastEl);
        
        // Dismiss on click anywhere on the page
        const dismissOnClick = () => {
            this.dismiss(toastEl);
            document.removeEventListener('click', dismissOnClick);
        };
        
        // Add click listener after a brief delay to avoid immediate dismissal
        setTimeout(() => {
            document.addEventListener('click', dismissOnClick, { once: true });
        }, 100);
        
        // Auto dismiss
        if (duration > 0) {
            setTimeout(() => {
                this.dismiss(toastEl);
            }, duration);
        }
        
        return toastEl;
    },
    
    dismiss(toastEl) {
        if (!toastEl || toastEl.classList.contains('toast-exit')) return;
        toastEl.classList.add('toast-exit');
        setTimeout(() => {
            toastEl.remove();
        }, 250);
    },
    
    dismissAll() {
        if (!this.container) return;
        const toasts = this.container.querySelectorAll('.toast');
        toasts.forEach(t => t.remove());
    },
    
    warning(title, message, duration) {
        return this.show({ title, message, duration });
    },
    
    error(title, message, duration) {
        return this.show({ title, message, duration });
    }
};

export { toast };

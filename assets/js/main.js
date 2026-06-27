/**
 * ============================================================
 * JAVASCRIPT FOR HYBRID AI FRAMEWORK WEBSITE
 * ============================================================
 */

document.addEventListener('DOMContentLoaded', function() {

    // ============================================================
    // 1. SCROLL ANIMATIONS (Fade-in effect)
    // ============================================================
    const animateOnScroll = () => {
        const elements = document.querySelectorAll('.feature-card, .stat-item, .highlight-box, .figure-container');
        
        const observer = new IntersectionObserver((entries) => {
            entries.forEach((entry, index) => {
                if (entry.isIntersecting) {
                    // Add a small delay for a cascading effect
                    setTimeout(() => {
                        entry.target.style.opacity = '1';
                        entry.target.style.transform = 'translateY(0)';
                    }, index * 80);
                    observer.unobserve(entry.target);
                }
            });
        }, {
            threshold: 0.15,
            rootMargin: '0px 0px -50px 0px'
        });

        elements.forEach(el => {
            el.style.opacity = '0';
            el.style.transform = 'translateY(25px)';
            el.style.transition = 'opacity 0.7s ease, transform 0.7s ease';
            observer.observe(el);
        });
    };

    // ============================================================
    // 2. CODE COPY FUNCTIONALITY
    // ============================================================
    const setupCodeCopy = () => {
        const codeBlocks = document.querySelectorAll('.code-block');
        
        codeBlocks.forEach(block => {
            // Create copy button
            const copyBtn = document.createElement('button');
            copyBtn.innerHTML = '<i class="fas fa-copy"></i> Copy';
            copyBtn.className = 'copy-btn';
            copyBtn.style.cssText = `
                position: absolute;
                top: 10px;
                right: 10px;
                background: #334155;
                color: white;
                border: none;
                padding: 0.3rem 0.8rem;
                border-radius: 6px;
                font-size: 0.75rem;
                cursor: pointer;
                opacity: 0.6;
                transition: opacity 0.2s, background 0.2s;
                font-family: 'Inter', sans-serif;
                z-index: 10;
            `;
            
            // Make block position relative for button positioning
            block.style.position = 'relative';
            block.appendChild(copyBtn);
            
            // Copy functionality
            copyBtn.addEventListener('click', () => {
                // Get text content without the button text
                const codeText = block.textContent.replace('Copy', '').trim();
                navigator.clipboard.writeText(codeText).then(() => {
                    copyBtn.innerHTML = '<i class="fas fa-check"></i> Copied!';
                    copyBtn.style.background = '#16a34a';
                    copyBtn.style.opacity = '1';
                    
                    setTimeout(() => {
                        copyBtn.innerHTML = '<i class="fas fa-copy"></i> Copy';
                        copyBtn.style.background = '#334155';
                        copyBtn.style.opacity = '0.6';
                    }, 2000);
                }).catch(() => {
                    // Fallback for older browsers
                    const textArea = document.createElement('textarea');
                    textArea.value = codeText;
                    document.body.appendChild(textArea);
                    textArea.select();
                    document.execCommand('copy');
                    document.body.removeChild(textArea);
                    
                    copyBtn.innerHTML = '<i class="fas fa-check"></i> Copied!';
                    copyBtn.style.background = '#16a34a';
                    setTimeout(() => {
                        copyBtn.innerHTML = '<i class="fas fa-copy"></i> Copy';
                        copyBtn.style.background = '#334155';
                    }, 2000);
                });
            });
            
            // Hover effect
            block.addEventListener('mouseenter', () => {
                copyBtn.style.opacity = '1';
            });
            block.addEventListener('mouseleave', () => {
                if (copyBtn.innerHTML.includes('Copy')) {
                    copyBtn.style.opacity = '0.6';
                }
            });
        });
    };

    // ============================================================
    // 3. AUTO-UPDATE FOOTER YEAR
    // ============================================================
    const updateFooterYear = () => {
        const yearSpan = document.getElementById('current-year');
        if (yearSpan) {
            yearSpan.textContent = new Date().getFullYear();
        }
    };

    // ============================================================
    // 4. SMOOTH SCROLL FOR INTERNAL LINKS (if any)
    // ============================================================
    const setupSmoothScroll = () => {
        document.querySelectorAll('a[href^="#"]').forEach(anchor => {
            anchor.addEventListener('click', function(e) {
                const targetId = this.getAttribute('href');
                if (targetId === '#') return;
                
                const target = document.querySelector(targetId);
                if (target) {
                    e.preventDefault();
                    target.scrollIntoView({
                        behavior: 'smooth',
                        block: 'start'
                    });
                }
            });
        });
    };

    // ============================================================
    // 5. INITIALIZE ALL FUNCTIONS
    // ============================================================
    animateOnScroll();
    setupCodeCopy();
    updateFooterYear();
    setupSmoothScroll();

    console.log('✅ Hybrid AI Framework website initialized successfully.');
});

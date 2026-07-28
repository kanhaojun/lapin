(function () {
  const STORAGE_KEY = 'lapin-lang';
  const html = document.documentElement;

  function setLang(lang) {
    html.setAttribute('data-lang', lang);
    localStorage.setItem(STORAGE_KEY, lang);
    document.querySelectorAll('.lang-btn').forEach((btn) => {
      btn.classList.toggle('active', btn.dataset.lang === lang);
    });
    document.title = lang === 'zh'
      ? 'Lapin — 工業影像分割框架'
      : 'Lapin — Industrial Image Segmentation';
  }

  const saved = localStorage.getItem(STORAGE_KEY);
  const browserZh = navigator.language.startsWith('zh');
  setLang(saved || (browserZh ? 'zh' : 'en'));

  document.querySelectorAll('.lang-btn').forEach((btn) => {
    btn.addEventListener('click', () => setLang(btn.dataset.lang));
  });

  const nav = document.querySelector('.nav');
  window.addEventListener('scroll', () => {
    nav.style.boxShadow = window.scrollY > 10
      ? '0 4px 24px rgba(0,0,0,0.3)'
      : 'none';
  });
})();

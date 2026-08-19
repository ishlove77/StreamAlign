/* ------------------------------------------------------------------
 * Compact audio player — replaces native <audio controls> with a
 * play/pause button, a seekable progress bar, and a time readout,
 * all within the narrow column width so the time stays visible.
 * ------------------------------------------------------------------ */
(function () {
  'use strict';

  var PLAY = '▶';   // ▶
  var PAUSE = '❚❚'; // ❚❚

  function fmt(t) {
    if (!isFinite(t) || t < 0) t = 0;
    var m = Math.floor(t / 60);
    var s = Math.floor(t % 60);
    return m + ':' + (s < 10 ? '0' : '') + s;
  }

  function enhance(audio) {
    audio.removeAttribute('controls');
    audio.style.display = 'none';

    var wrap = document.createElement('div');
    wrap.className = 'audio-player';

    var btn = document.createElement('button');
    btn.type = 'button';
    btn.className = 'ap-btn';
    btn.textContent = PLAY;
    btn.setAttribute('aria-label', 'Play');

    var bar = document.createElement('div');
    bar.className = 'ap-bar';
    var fill = document.createElement('div');
    fill.className = 'ap-fill';
    bar.appendChild(fill);

    var time = document.createElement('span');
    time.className = 'ap-time';
    time.textContent = '0:00';

    wrap.appendChild(btn);
    wrap.appendChild(bar);
    wrap.appendChild(time);
    audio.parentNode.insertBefore(wrap, audio);

    function render() {
      var dur = audio.duration;
      var cur = audio.currentTime;
      if (isFinite(dur) && dur > 0) {
        fill.style.width = (cur / dur * 100) + '%';
        time.textContent = fmt(cur) + '/' + fmt(dur);
      } else {
        time.textContent = fmt(cur);
      }
    }

    btn.addEventListener('click', function () {
      if (audio.paused) {
        // Pause any other player so comparisons play one at a time.
        document.querySelectorAll('.audio-table audio').forEach(function (a) {
          if (a !== audio) a.pause();
        });
        audio.play();
      } else {
        audio.pause();
      }
    });

    bar.addEventListener('click', function (e) {
      var dur = audio.duration;
      if (!isFinite(dur) || dur <= 0) return;
      var rect = bar.getBoundingClientRect();
      var ratio = (e.clientX - rect.left) / rect.width;
      ratio = Math.max(0, Math.min(1, ratio));
      audio.currentTime = ratio * dur;
    });

    audio.addEventListener('play', function () {
      btn.textContent = PAUSE;
      btn.setAttribute('aria-label', 'Pause');
    });
    audio.addEventListener('pause', function () {
      btn.textContent = PLAY;
      btn.setAttribute('aria-label', 'Play');
    });
    audio.addEventListener('ended', function () {
      btn.textContent = PLAY;
      btn.setAttribute('aria-label', 'Play');
    });
    audio.addEventListener('loadedmetadata', render);
    audio.addEventListener('timeupdate', render);
  }

  function init() {
    document.querySelectorAll('.audio-table audio').forEach(enhance);
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
  } else {
    init();
  }
})();

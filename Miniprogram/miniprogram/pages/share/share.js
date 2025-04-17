// pages/share/share.js
Page({
  data: {
    modellist: ['拾穗者', '画家与模特', '镜前的少女', '星月夜', 'Udnie', '神奈川冲浪里', 'SDUWH'],
  },

  onLoad: function (options) {
    this.setData({
      imgsrc: options.imgsrc,
      style: options.style,
    })
  }
})
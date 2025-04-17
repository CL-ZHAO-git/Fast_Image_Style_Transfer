const db = wx.cloud.database({});
const cont = db.collection('history');

Page({
  data: {
    openid: null,
    list:[],
    modellist: ['拾穗者', '画家与模特', '镜前的少女', '星月夜', 'Udnie', '神奈川冲浪里', 'SDUWH'],
  },

  onLoad: function (options) {
    var _this=this
    var openid=wx.getStorageSync("userInfo");
    this.setData({
      openid:openid
    })
    db.collection('history').where({
      _openid: openid,
    }).get({
      success: function(res) {
        _this.setData({
          list: res.data,
        })
      }
    })
  }
})
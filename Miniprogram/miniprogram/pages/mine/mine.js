// miniprogram/pages/mine/mine.js
var app = getApp()

Page({
  data: {
    userInfo: {},
  },

  onLoad: function (e) {
    wx.cloud.callFunction({
      name: 'index',
      data: {},
      success: res => {
        this.setData({
          userInfo: res.result.openid,
        })
        wx.setStorageSync('userInfo', res.result.openid)
      }
    })
  },

  gotoours: function () {
    wx.navigateTo({
      url: '../ours/ours',
    })
  },

  gotohistory: function () {
    let userInfo = this.data.userInfo
    wx.navigateTo({
      url: '../history/history?openid=userInfo',
    })
  }
})
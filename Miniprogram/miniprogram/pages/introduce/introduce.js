var faData = require("../../data/family.js");

Page({
  data: {},
 
  onLoad: function (options) {
    this.setData({ familyKey: faData.familys });
  },

  toDetail: function (event) {
    var fid = event.currentTarget.dataset.familyId;
    wx.navigateTo({
      url: 'pic'+fid+'/pic' + fid
    })
  }
})
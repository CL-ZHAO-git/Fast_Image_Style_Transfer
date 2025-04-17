import Notify from '../../@vant/weapp/notify/notify';
var app = getApp();
const db = wx.cloud.database()
const _ = db.command

Page({
  /**
   * 页面的初始数据
   */
  data: {
    text:'风格简介',
    list: [
      {num:1,name:'拾穗者',proUrl:'/images/style/des_glaneuses.jpg', choose:false, generate:false},
      {num:2,name:'画家与模特',proUrl:'/images/style/la_muse.jpg', choose:false, generate:false},
      {num:3,name:'镜前的少女',proUrl: '/images/style/mirror.jpg', choose:false, generate:false},
      {num:4,name:'星月夜',proUrl:'/images/style/starry_night.jpg',choose:false, generate:false}, 
      {num:5,name:'Udnie',proUrl:'/images/style/udnie.jpg',choose:false, generate:false},
      {num:6,name:'神奈川冲浪里',proUrl:'/images/style/wave_crop.jpg',choose:false, generate:false},
      {num:7,name:'SDUWH',proUrl:'/images/style/sduwh.jpg',choose:false, generate:false}
    ],
    current: 0,
    animationData: {},
    animationData2: {},
    changeStyle: false,
    choose: false,
    show: false,
    showShare: false,
    options: [
      { name: '微信', icon: 'wechat', openType: 'share' },
      { name: 'QQ', icon: 'qq' },
      { name: '保存图片', icon: '/images/img1.png' },
      { name: '分享海报', icon: 'poster' },
    ],
    modellist: ['des_glaneuses', 'la_muse', 'mirror', 'starry_night', 'udnie', 'wave_crop', 'sduwh']
  },

  onLoad: function (options) {
    this.stretch(370)
    this.shrink(350)
  },
  
  change(e){
    this.setData({
      current: e.detail.current
    })
    this.stretch(370)
    this.shrink(350)
  },

  // 收缩
  stretch(h){
    var animation = wx.createAnimation({
      duration: 250,
      timingFunction: 'ease',
    })
    this.animation = animation
    animation.height(h).step()
    this.setData({
      animationData: animation.export(),
    })
  },

  // 展开
  shrink(h){
    var animation2 = wx.createAnimation({
      duration: 250,
      timingFunction: 'ease',
    })
    this.animation2 = animation2
    animation2.height(h).step()
    this.setData({
      animationData2: animation2.export()
    })
  },

  goToDescription1(){
    wx.navigateTo({
      url: '../introduce/pic1/pic1',
    })
  },

  goToDescription2(){
    wx.navigateTo({
      url: '../introduce/pic2/pic2',
    })
  },

  goToDescription3(){
    wx.navigateTo({
      url: '../introduce/pic3/pic3',
    })
  },

  goToDescription4(){
    wx.navigateTo({
      url: '../introduce/pic4/pic4',
    })
  },

  goToDescription5(){
    wx.navigateTo({
      url: '../introduce/pic5/pic5',
    })
  },

  goToDescription6(){
    wx.navigateTo({
      url: '../introduce/pic6/pic6',
    })
  },

  goToDescription7(){

  },

  chooseimg: function() {
    var _this = this
    wx.chooseImage({
      success (res) {
        var temp1 = 'list[' + _this.data.current + '].proUrl'
        var temp2 = 'list[' + _this.data.current + '].tempimg'
        var temp3 = 'list[' + _this.data.current + '].choose'
        _this.setData({
          [temp1]: res.tempFilePaths[0],
          [temp2]: res.tempFilePaths[0],
          [temp3]: true,
        })
        Notify({
          type: 'success',
          message: '选择成功',
          duration: 1000,
        });
      }
    })
  },

  generate: function() {
    var _this = this
    if(this.data.list[_this.data.current].choose==false){ 
      wx.showToast({
        title: '请选择图片',
        icon: 'none',
      })
    }else{
      var numvalue = _this.data.current
      wx.showLoading({
        title: '生成图片中...',
      })
      _this.setData({
        changeStyle: true,
      })
      wx.uploadFile({
        url: 'https://experimentforzcl.cn:8080',
        filePath: _this.data.list[_this.data.current].tempimg,
        name: 'file',
        formData:{'style': _this.data.modellist[_this.data.current],
            'original_color': Number(_this.data.original_color),
            'blend_alpha': Number(_this.data.blend_alpha)
          },
        success (res){
          if(res.statusCode==200){
            var temp1 = 'list[' + numvalue + '].proUrl'
            var temp2 = 'list[' + numvalue + '].generate'
            _this.setData({
              [temp1]: "data:image/png;base64," + res.data,
              [temp2]: true,
              changeStyle: false,
            })

            var number = Math.random();
            var timestamp = Date.parse(new Date());
            var date = new Date(timestamp);
            var time = date.getFullYear() + '-' + (date.getMonth() + 1 < 10 ? '0' + (date.getMonth() + 1) : date.getMonth() + 1) + '-' + (date.getDate() < 10 ? '0' + date.getDate() : date.getDate()) + "\t" + (date.getHours() < 10 ? '0' + date.getHours() : date.getHours()) + ":" + (date.getMinutes() < 10 ? '0' + date.getMinutes() : date.getMinutes())

            wx.getFileSystemManager().writeFile({
              filePath: wx.env.USER_DATA_PATH + '/pic' + number + '.png',
              data: _this.data.list[numvalue].proUrl.slice(22),
              encoding: 'base64',
            })
            wx.cloud.uploadFile({
              cloudPath: 'history/'+number+'.png',
              filePath: wx.env.USER_DATA_PATH + '/pic' + number + '.png',
              success: res => {
                db.collection('history').add({
                  data: {
                    time: time,
                    imgsrc: 'cloud://charlie-9mgmr.6368-charlie-9mgmr-1301103640/history/'+number+'.png',
                    style: numvalue,
                  }
                })
              },
            })
            wx.hideLoading(),
            Notify({
              type: 'success',
              message: '生成成功',
              duration: 1000,
            });
          }
        }
      })
    }
  },

  changeColor: function(res) {
    this.setData({
      original_color: res.detail/100
    })
    wx.showToast({
      icon: 'none',
      title: `颜色保留：${res.detail}%`,
    });
  },

  styleChange: function(res) {
    this.setData({
      blend_alpha: res.detail/100
    })
    wx.showToast({
      icon: 'none',
      title: `风格占比：${res.detail}%`,
    });
  },

  gotoShare: function() {
    this.setData({
      showShare: true,
    })
  },

  shareSelect(res) {
    var _this = this
    if(res.detail.index==2){
      if(_this.data.list[_this.data.current].generate==true){
        var number = Math.random();
        wx.getFileSystemManager().writeFile({
          filePath: wx.env.USER_DATA_PATH + '/pic' + number + '.png',
          data: _this.data.list[_this.data.current].proUrl.slice(22),
          encoding: 'base64',
          success: res => {
            wx.saveImageToPhotosAlbum({
              filePath: wx.env.USER_DATA_PATH + '/pic' + number + '.png',
              success: function (res) {
                wx.showToast({
                  title: '保存成功!',
                  icon: "none"
                })
              },
              fail: function (err) {
                wx.showToast({
                  title: '保存失败',
                  icon: "none"
                })
              }
            })
          },
        })
      }else{
        wx.showToast({
          title: '保存失败，请先生成图片',
          icon: 'none'
        })
      }
    }else if(res.detail.index==3){
      if(_this.data.list[_this.data.current].generate==true){
        var number = Math.random();
        wx.getFileSystemManager().writeFile({
          filePath: wx.env.USER_DATA_PATH + '/pic' + number + '.png',
          data: _this.data.list[_this.data.current].proUrl.slice(22),
          encoding: 'base64'
        })
        wx.navigateTo({
          url: '/pages/share/share?imgsrc=' + wx.env.USER_DATA_PATH + '/pic' + number + '.png'+'&style='+_this.data.current,
        })
      }else{
        wx.showToast({
          title: '保存失败，请先生成图片',
          icon: 'none'
        })
      }
    }
    this.shareClose();
  },

  shareClose() {
    this.setData({
      showShare: false
    });
  }
})
import numpy as np
import cv2
import random
import math

class Game():
    # 初期設定
    def __init__(self, isDebugmode=False):
        self.isDebugmode = isDebugmode

        # 計算途中でself.~~が並ぶと見づらいのでいったんローカル変数で処理する
        filename = "earth.png"
        earth_img = cv2.imread(filename)
        r = earth_img.shape[0] // 2
        w, h = 1024, 768
        xc, yc = w//2, h//2
        image_origin = np.full((h,w,3), (0,0,0), np.uint8)
        image_origin[yc-r:yc+r, xc-r:xc+r] = earth_img
        if self.isDebugmode:
            cv2.circle(image_origin, (xc,yc), r, (0,0,255), 1)

        # あらためてインスタンス定数を定義する
        self.winname = "meteor game"
        self.width, self.height = w, h
        self.xc, self.yc = xc, yc
        self.image_origin = image_origin
        self.radius = r
        self.meteor_num = 0
        self.meteors = []
        self.beam = Beam(self)
        self.damage = 0
        self.score = 0
        self.isGameover = False
        self.next_score = 0

    # ゲーム画面描写
    def show(self):
        self.image = self.image_origin.copy()

        # 隕石を描く
        for meteor in self.meteors:
            if self.isDebugmode:
                self.draw_meteor_track(meteor)
            self.draw_meteor(meteor)

        # ビームもしくは爆風を描く
        if self.beam.isShooting:
            self.draw_beam(self.beam)

        # スコアを描く
        text = f"DAMAGE:{self.damage:3d}     SCORE:{self.score:5d}"
        cv2.putText(self.image, text, (20,50), cv2.FONT_HERSHEY_DUPLEX, 1, (255,255,255), 1)
        cv2.imshow(self.winname, self.image)
    
        # ビーム発射後の処理
        self.beam.cool_down()

        # 隕石消滅処理　ループ内でリストの要素を削除するため、リストでなくリストのコピーでループを回す
        for meteor in self.meteors[:]:
            if meteor.isCrash and meteor.timer > 10:
                self.meteors.remove(meteor)


    # 隕石を描く
    def draw_meteor(self, meteor):
        x, y = meteor.pos
        if meteor.isCrash:
            meteor.timer += 1
            r = meteor.timer * 10
            cv2.circle(self.image, (int(x), int(y)), r, (0,0,255), -1)
        else:
            home = (meteor.radius, meteor.radius)
            self.image = putSprite(self.image, meteor.image, (x,y), meteor.angle, home)


    # 隕石の軌跡を描く
    def draw_meteor_track(self, meteor):
        for tm in range(0, meteor.max_cnt, 50):
            # 制御点数3のベジェ曲線を描く
            t = tm / meteor.max_cnt                         # 0<=t<=1 に正規化
            x = int((1-t)**2 * meteor.x1 + 2 * (1-t) * t * meteor.x2 + t**2 * meteor.x3)
            y = int((1-t)**2 * meteor.y1 + 2 * (1-t) * t * meteor.y2 + t**2 * meteor.y3)
            cv2.circle(self.image, (x,y), 1, (255,255,255), -1)


    # ビームを描く
    def draw_beam(self, beam):
        # 最初の数フレームはビームを描く
        if beam.timer < 10:
            cv2.line(self.image, beam.pos0, beam.pos1, (0,255,255), 2)

        # 爆風を描く
        r = int(beam.radius)
        cv2.circle(self.image, beam.pos0, r, (255,255,255), -1)

        # 当たり判定
        xb, yb = beam.pos0
        for meteor in self.meteors:
            xm, ym = meteor.pos
            if (xb-xm)**2+(yb-ym)**2 < (beam.radius + meteor.radius)**2:
                meteor.isCrash = True
                self.beam.isHitMeteor = True
                self.score += 10

        # 地球に当たってしまったらダメージ　ただし隕石撃墜できていたらセーフ
        if not self.beam.isHitMeteor and not self.beam.isHitEarth:
            if (xb-self.xc)**2+(yb-self.yc)**2 < (self.radius+self.beam.radius)**2:
                self.damage += 10
                self.beam.isHitEarth = True

    # ゲームオーバー処理
    def gameover(self):
        gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        img = cv2.merge((gray, gray, gray))

        # GAME OVERの文字を少しずつ赤くする
        for c in range(255):
            img = putText_center(img, "GAME OVER", cv2.FONT_HERSHEY_COMPLEX, 5, (0,0,c), 5)
            cv2.imshow(self.winname, img)
            cv2.waitKey(10)
        cv2.waitKey(1000)
        cv2.destroyAllWindows()


    # ゲーム画面上でマウスを動かしたときの処理
    def move_mouse(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.beam.shoot(x, y)


class Beam():
    # 初期設定
    def __init__(self, game):
        self.pos0 = (0,0)
        self.pos1 = (0,0)
        self.length = game.width + game.height              # ビームの長さ（仮）
        self.max_cnt = 100
        self.initialize()
        
    # ビームの初期化
    def initialize(self):
        self.isShooting = False
        self.isHitMeteor = False
        self.isHitEarth = False
        self.timer = 0
        self.radius = 10

    # 発射
    def shoot(self, x, y):
        if not self.isShooting:
            self.isShooting = True
            self.pos0 = (x, y)                              # マウスポインタの座標
            angle = random.random() * 2 * math.pi           # ランダムな角度
            x1 = int(self.length * math.cos(angle) + x)     # ビームの根元座標
            y1 = int(self.length * math.sin(angle) + y)     # 　長さ（仮）が対角線より長いので
            self.pos1 = (x1, y1)                            # 　必ず画面外になる、はず。

    # 発射後の演出
    def cool_down(self):
        if self.isShooting:
            self.timer += 1
            self.radius += 0.2
            if self.timer > self.max_cnt:
                self.initialize()


class Meteor():
    # 初期設定
    def __init__(self, game):
        rnd = random.randint(1, 4)                          # 隕石画像の数をカウントする処理は未実装
        filename = f"meteor{rnd}.png"
        img = cv2.imread(filename, -1)
        size = max(20, int(img.shape[0] * (1 - game.score/1000)))
        img = cv2.resize(img, (size,size))
        r = img.shape[0] // 2
        self.image = img
        if game.isDebugmode:
            cv2.circle(self.image, (r, r), r, (0,0,255,255), 1)
            cv2.circle(self.image, (r, r), 10, (0,0,255,255), -1)
        self.radius = r
        self.angle = random.randint(0, 359)
        self.plus_angle = 2 * random.random() - 1
        self.isCrash = False
        self.timer = 0
        self.max_cnt = max(500,int(1000 - game.score/10))
        self.x1, self.y1 = getP1(game, self.radius)
        self.x3, self.y3 = getP3(game)
        self.x2, self.y2 = getP2(game)
        self.pos = (self.x1, self.y1)

    def move(self, game):
        self.timer += 1
        self.angle += self.plus_angle
        # 制御点数3のベジェ曲線を描く
        t = self.timer / self.max_cnt                       # 0<=t<=1 に正規化
        x = int((1-t)**2 * self.x1 + 2 * (1-t) * t * self.x2 + t**2 * self.x3)
        y = int((1-t)**2 * self.y1 + 2 * (1-t) * t * self.y2 + t**2 * self.y3)
        self.pos = (x, y)

        # 地球との衝突判定
        if (x-game.xc)**2 + (y-game.yc)**2 < game.radius**2:
            self.isCrash = True
            game.damage += 20
            self.timer = 0                                  # カウンターをリセット（爆発で再利用する）


# 画面外周のどこかを選ぶ　正確には外周ではなく画面の少し外側
# もう少しスマートに書けないものか
def getP1(game, offset = 0):
    min_x = -offset
    max_x = game.width + offset -1
    min_y = -offset
    max_y = game.height + offset -1
    x = random.randint(min_x, max_x)
    y = random.randint(min_y, max_y)
    rnd = random.randint(1,4)
    if rnd == 1:
        y = min_y
    elif rnd == 2:
        y = max_y
    elif rnd == 3:
        x = min_x
    else:
        x = max_x
    return (x, y)


# 画面内のどこかを選ぶ
def getP2(game):
    x = random.randint(0, game.width)
    y = random.randint(0, game.height)
    return (x, y)

# 地球内部のどこかを選ぶ
def getP3(game):
    r = random.random() * game.radius
    a = random.random() * 2 * math.pi
    x = int(game.xc + r * math.cos(a))
    y = int(game.yc + r * math.sin(a))
    return (x, y)


# 自作スプライト関数　https://qiita.com/mo256man/items/d89c53a39c8e82b9c889
def putSprite(back, front4, pos, angle=0, home=(0,0)):
    fh, fw = front4.shape[:2]
    bh, bw = back.shape[:2]
    x, y = pos
    xc, yc = home[0] - fw/2, home[1] - fh/2
    a = np.radians(angle)
    cos , sin = np.cos(a), np.sin(a)
    w_rot = int(fw * abs(cos) + fh * abs(sin))
    h_rot = int(fw * abs(sin) + fh * abs(cos))
    M = cv2.getRotationMatrix2D((fw/2,fh/2), angle, 1)
    M[0][2] += w_rot/2 - fw/2
    M[1][2] += h_rot/2 - fh/2
    imgRot = cv2.warpAffine(front4, M, (w_rot,h_rot))
    xc_rot = xc * cos + yc * sin
    yc_rot = -xc * sin + yc * cos
    x0 = int(x - xc_rot - w_rot / 2)
    y0 = int(y - yc_rot - h_rot / 2)
    if not ((-w_rot < x0 < bw) and (-h_rot < y0 < bh)):
        return back
    x1, y1 = max(x0, 0), max(y0,  0)
    x2, y2 = min(x0+w_rot, bw), min(y0+h_rot, bh)
    imgRot = imgRot[y1-y0:y2-y0, x1-x0:x2-x0]
    result = back.copy()
    front = imgRot[:, :, :3]
    mask1 = imgRot[:, :, 3]
    mask = 255 - cv2.merge((mask1, mask1, mask1))
    roi = result[y1:y2, x1:x2]
    tmp = cv2.bitwise_and(roi, mask)
    tmp = cv2.bitwise_or(tmp, front)
    result[y1:y2, x1:x2] = tmp
    return result


# 画面中央に文字を描写する
def putText_center(img, text, fontFace, fontScale, color, thickness):
    (tw, th), _ = cv2.getTextSize(text, fontFace, fontScale, thickness)
    yc, xc = img.shape[0] // 2, img.shape[1] // 2
    x , y = xc-tw//2, yc+th//2
    img = cv2.putText(img, text, (x, y), fontFace, fontScale, color, thickness) 
    return img


def main():
    game = Game()
    cv2.namedWindow(game.winname)
    cv2.setMouseCallback(game.winname, game.move_mouse)
    while not game.isGameover:
        # 一定スコアごとに隕石の最大出現数が増える
        if game.score >= game.next_score:
            game.meteor_num += 1
            game.next_score += 100

        # 1%の確率で隕石が出現する　もちろん現在の数が最大出現数未満のときのみ
        if len(game.meteors) < game.meteor_num and random.random() < 0.01:
            game.meteors.append(Meteor(game))

        # 隕石を動かす
        for meteor in game.meteors:
            if not meteor.isCrash:
                meteor.move(game)
        
        game.show()

        # ダメージ100でゲームオーバー
        if game.damage >= 100:
            game.isGameover = True

        key = cv2.waitKey(1) & 0xFF
        if key == 27:
            break

    game.gameover()

if __name__ == "__main__":
    main()

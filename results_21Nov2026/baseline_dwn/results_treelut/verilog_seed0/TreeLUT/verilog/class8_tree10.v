module class8_tree10(input wire [467:0] i, output wire [0:0] o);

reg [0:0] o_reg;
assign o = o_reg;

always @(*) begin
	if((i[235] & (~i[244]) & (~i[177])) | ((~i[196]) & (~i[416]) & (~i[154]) & i[177]) | (i[185] & (~i[107]) & i[432] & i[154] & i[177]) | (i[355] & i[167] & (~i[432]) & i[154] & i[177]) | ((~i[275]) & (~i[167]) & (~i[432]) & i[154] & i[177]) | (i[399] & (~i[296]) & i[416] & (~i[154]) & i[177]))
		o_reg = 0;
	else
		o_reg = 1;
end

endmodule

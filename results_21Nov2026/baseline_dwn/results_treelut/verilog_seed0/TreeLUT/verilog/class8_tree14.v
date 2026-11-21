module class8_tree14(input wire [467:0] i, output wire [0:0] o);

reg [0:0] o_reg;
assign o = o_reg;

always @(*) begin
	if((i[113] & i[61] & (~i[224])) | (i[248] & (~i[203]) & i[326] & i[224]) | (i[289] & i[95] & (~i[326]) & i[224]) | ((~i[171]) & i[183] & (~i[61]) & (~i[224])) | (i[251] & (~i[183]) & (~i[61]) & (~i[224])) | (i[163] & (~i[227]) & i[203] & i[326] & i[224]) | ((~i[321]) & (~i[251]) & (~i[183]) & (~i[61]) & (~i[224])))
		o_reg = 0;
	else
		o_reg = 1;
end

endmodule
